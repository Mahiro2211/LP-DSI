"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import os
import time
import json
import datetime
import math

import torch

from ..misc import dist_utils, stats

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler


def _state_to_cpu(state):
    """Deep-copy a solver state dict to CPU. state_dict() tensors share storage
    with the live GPU modules, so without the copy a cached "best" snapshot
    would mutate as training keeps updating the parameters in place."""
    def _to_cpu(obj):
        if torch.is_tensor(obj):
            return obj.detach().to('cpu', copy=True)
        if isinstance(obj, dict):
            return {k: _to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_to_cpu(v) for v in obj)
        return obj
    return _to_cpu(state)


class DetSolver(BaseSolver):

    def fit(self, ):
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches,
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch)
            self.self_lr_scheduler = True
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        # W&B experiment tracking (soft dep: skipped if wandb not installed).
        # Initialized once here on the main process; finished at the end of fit().
        self._wandb_init(n_parameters)

        # Checkpoint opt-out (--no-ckpt): log.txt / eval/ / TensorBoard / W&B
        # are unaffected; only the .pth weight files are skipped. The stage-1
        # best snapshot the stage-2 restart and EMA rollback reload is then
        # kept in CPU RAM instead of on disk (see _save_best_stg1).
        yaml_cfg = self.cfg.yaml_cfg if hasattr(self.cfg, 'yaml_cfg') else {}
        self._save_ckpt = yaml_cfg.get('save_ckpt', True)
        self._best_stg1_state = None

        top1 = 0
        best_stat = {'epoch': -1, }
        # Stage-2 EMA-search rollback gating. Defaults read from
        # BatchImageCollateFunction reproduce the legacy behavior exactly:
        # refresh after a single non-improving epoch, no cooldown, no decay
        # floor, unlimited refreshes.
        _collate = self.train_dataloader.collate_fn
        ema_search_enabled = getattr(_collate, 'ema_search_enabled', True)
        ema_search_patience = getattr(_collate, 'ema_search_patience', 1)
        ema_search_cooldown = getattr(_collate, 'ema_search_cooldown', 0)
        ema_search_decay_step = getattr(_collate, 'ema_search_decay_step', 0.0001)
        ema_search_min_decay = getattr(_collate, 'ema_search_min_decay', 0.0)
        ema_search_max_refreshes = getattr(_collate, 'ema_search_max_refreshes', float('inf'))
        no_improve = 0
        last_refresh = -float('inf')
        refresh_count = 0
        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )
            for k in test_stats:
                # rank by AP50 (coco stats[1]), not mAP@[.5:.95] (stats[0])
                ap50 = test_stats[k][1]
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = ap50
                top1 = ap50
                print(f'best_stat: {best_stat}')

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):
            epoch_t0 = time.time()

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self._load_best_stg1()
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats, grad_percentages = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                teacher_model=self.teacher_model, # NEW: Pass teacher model to train_one_epoch
            )

            if not self.self_lr_scheduler:  # update by epoch 
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1
            if dist_utils.is_main_process() and hasattr(self.criterion, 'distill_adaptive_params') and \
                self.criterion.distill_adaptive_params and self.criterion.distill_adaptive_params.get('enabled', False):

                params = self.criterion.distill_adaptive_params
                default_weight = params.get('default_weight')

                avg_percentage = sum(grad_percentages) / len(grad_percentages) if grad_percentages else 0.0

                current_weight = self.criterion.weight_dict.get('loss_distill', 0.0)
                new_weight = current_weight
                reason = 'unchanged'

                if avg_percentage < 1e-6:
                    if default_weight is not None:
                        new_weight = default_weight
                        reason = 'reset_to_default_zero_grad'
                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    if default_weight is not None:
                        new_weight = default_weight
                        reason = 'ema_phase_default'
                else:
                    rho = params['rho']
                    delta = params['delta']
                    lower_bound = rho - delta
                    upper_bound = rho + delta
                    if not (lower_bound <= avg_percentage <= upper_bound):
                        target_percentage = upper_bound if avg_percentage < lower_bound else lower_bound
                        if current_weight > 1e-6:
                            p_current = avg_percentage / 100.0
                            p_target = target_percentage / 100.0
                            numerator = p_target * (1.0 - p_current)
                            denominator = p_current * (1.0 - p_target)
                            if abs(denominator) >= 1e-9:
                                ratio = numerator / denominator
                                ratio = max(ratio, 0.1)  # clamp non-positive to 0.1
                                new_weight = current_weight * ratio
                                new_weight = min(max(new_weight, current_weight / 10.0), current_weight * 10.0)
                                reason = f'adjusted_to_{target_percentage:.2f}%'

                if abs(new_weight - current_weight) > 0:
                    self.criterion.weight_dict['loss_distill'] = new_weight
                print(f"Epoch {epoch}: avg encoder grad {avg_percentage:.2f}% | distill {current_weight:.6f} -> {new_weight:.6f} ({reason})")

            # Keep a single rolling checkpoint for resume; periodic
            # checkpoint{NNNN}.pth snapshots are no longer written because
            # they filled the disk on long runs. Best weights are saved by
            # AP50 below (best_stg1.pth / best_stg2.pth). Saved on every
            # epoch, including after stop_epoch, so resume stays current.
            # Skipped in --no-ckpt runs: by design they cannot be resumed.
            if self.output_dir and self._save_ckpt:
                dist_utils.save_on_master(self.state_dict(), self.output_dir / 'last.pth')

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            # TODO
            for k in test_stats:
                # rank by AP50 (coco stats[1]), not mAP@[.5:.95] (stats[0])
                ap50 = test_stats[k][1]
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                if k in best_stat:
                    best_stat['epoch'] = epoch if ap50 > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], ap50)
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = ap50

                if best_stat[k] > top1:
                    best_stat_print['epoch'] = epoch
                    top1 = best_stat[k]
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            if self._save_ckpt:
                                dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                        else:
                            self._save_best_stg1()

                best_stat_print[k] = max(best_stat[k], top1)
                print(f'best_stat: {best_stat_print}')  # global best

                if best_stat['epoch'] == epoch and self.output_dir:
                    no_improve = 0
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if ap50 > top1:
                            top1 = ap50
                            if self._save_ckpt:
                                dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                    else:
                        top1 = max(ap50, top1)
                        self._save_best_stg1()

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    no_improve += 1
                    can_refresh = (
                        ema_search_enabled
                        and no_improve >= ema_search_patience
                        and epoch - last_refresh >= ema_search_cooldown
                        and self.ema.decay - ema_search_decay_step >= ema_search_min_decay
                        and refresh_count < ema_search_max_refreshes
                    )
                    if can_refresh:
                        n_stall = no_improve
                        best_stat = {'epoch': -1, }
                        self.ema.decay -= ema_search_decay_step
                        self._load_best_stg1()
                        no_improve = 0
                        last_refresh = epoch
                        refresh_count += 1
                        print(f'Refresh EMA at epoch {epoch} '
                              f'(no improve for {n_stall} epochs, refresh #{refresh_count}) '
                              f'with decay {self.ema.decay}')


            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

            # Push the same per-epoch metrics to W&B (no-op if disabled).
            self._wandb_log(train_stats, test_stats.get('coco_eval_bbox'), epoch,
                            time.time() - epoch_t0)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        self._wandb_finish()


    def val(self, ):
        self.eval()

        # W&B tracking is also useful in eval-only mode (logs the AP/AR once).
        self._wandb_init(None)

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        self._wandb_log(None, test_stats.get('coco_eval_bbox'), self.last_epoch, None)
        self._wandb_finish()

        return


    def _save_best_stg1(self):
        """Persist the stage-1 best weights: to disk normally, to CPU RAM
        under --no-ckpt. The stage-2 restart and the EMA-search rollback both
        reload this snapshot (see _load_best_stg1), so it must exist even when
        checkpoints are off. The RAM snapshot is taken on every rank, not just
        the master, because every rank performs the reload."""
        if not self.output_dir:
            return
        if self._save_ckpt:
            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')
        else:
            self._best_stg1_state = _state_to_cpu(self.state_dict())

    def _load_best_stg1(self):
        """Reload the stage-1 best weights (counterpart of _save_best_stg1)."""
        if self._save_ckpt:
            self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
            return
        if self._best_stg1_state is not None:
            self.load_state_dict(self._best_stg1_state)
            return
        # A run resumed from checkpoint-enabled training into --no-ckpt mode
        # has no RAM snapshot yet — fall back to the on-disk file if present.
        p = self.output_dir / 'best_stg1.pth'
        if p.exists():
            self.load_resume_state(str(p))
        else:
            raise RuntimeError(
                'cannot restart stage 2: this --no-ckpt run has neither a '
                f'cached stage-1 best snapshot nor {p} on disk')

    def state_dict(self):
        """State dict, train/eval"""
        state = {}
        state['date'] = datetime.datetime.now().isoformat()

        # For resume
        state['last_epoch'] = self.last_epoch
        # W&B run id so `-r` resume continues the same cloud run
        if getattr(self, '_wandb_run_id', None):
            state['wandb_run_id'] = self._wandb_run_id

        for k, v in self.__dict__.items():
            if k == 'teacher_model':
                continue
            if hasattr(v, 'state_dict'):
                v = dist_utils.de_parallel(v)
                state[k] = v.state_dict()

        return state

    # ------------------------------------------------------------------
    # W&B experiment tracking, ported from the DCEA project so runs from
    # both codebases land in the SAME dashboard with IDENTICAL metric keys
    # (test/AP ... test/ARl) and identical computation: both take the
    # standard 12-element COCOeval stats array (test_stats['coco_eval_bbox']).
    # ``import wandb`` is deferred to ``_wandb_init``; if it is not installed
    # (or not on the main process) every method becomes a no-op, so training
    # never breaks because of a missing optional dependency.
    # ------------------------------------------------------------------
    # W&B project ``sar_obj_detection`` under the ``wfu`` entity (same as
    # DCEA); override with the WANDB_ENTITY / WANDB_PROJECT env vars.
    # ``entity=`` and ``project=`` must be passed as SEPARATE args to
    # ``wandb.init`` — putting ``wfu/sar_obj_detection`` in ``project=``
    # raises "Invalid project name ... cannot contain '/'".
    _WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "wfu")
    _WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "sar_obj_detection")

    # Column names follow pycocotools coco_eval.stats order (length 12),
    # named in paper terms for AP & AR. Leading 'epoch' is not a metric.
    _APAR_HEADER = [
        "epoch",
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "AR1",
        "AR10",
        "AR100",
        "ARs",
        "ARm",
        "ARl",
    ]

    # Class-level default so _wandb_log/_wandb_finish are safe no-ops even
    # before _wandb_init runs; as a class attr it never enters state_dict().
    _wandb = None

    def _wandb_init(self, n_parameters):
        """Initialize a W&B run on the main process (no-op otherwise).

        Opt-in via the ``use_wandb`` config key (set by the launcher's
        ``--wandb`` flag). Off by default so W&B is never started unless the
        user explicitly asks for it. Even when enabled, a missing ``wandb``
        package or a network/auth failure degrades to a no-op.
        """
        self._wandb = None
        cfg = self.cfg.yaml_cfg if hasattr(self.cfg, 'yaml_cfg') else {}
        if not cfg.get('use_wandb', False):
            return
        if not dist_utils.is_main_process():
            return
        try:
            import wandb
        except ImportError:
            print("[wandb] not installed; skipping W&B logging "
                  "(pip install wandb to enable).")
            return
        if wandb.run is not None:
            # Already initialized (e.g. fit() called twice); reuse it.
            self._wandb = wandb
            return

        # Flatten the resolved YAML config into loggable scalars; pull the
        # common training knobs and a few model identifiers.
        config = {
            'task': cfg.get('task'),
            'model': cfg.get('model'),
            'criterion': cfg.get('criterion'),
            'num_classes': cfg.get('num_classes'),
            'epoches': cfg.get('epoches'),
            'use_amp': cfg.get('use_amp'),
            'use_ema': cfg.get('use_ema'),
            'clip_max_norm': cfg.get('clip_max_norm'),
            'n_parameters': n_parameters,
        }
        opt = cfg.get('optimizer') or {}
        if isinstance(opt, dict):
            config.update({
                'optimizer_type': opt.get('type'),
                'lr': opt.get('lr'),
                'weight_decay': opt.get('weight_decay'),
            })
        sched = cfg.get('lr_scheduler') or {}
        if isinstance(sched, dict):
            config['lr_scheduler'] = sched.get('type')
        tld = cfg.get('train_dataloader') or {}
        if isinstance(tld, dict):
            config['total_batch_size'] = tld.get('total_batch_size')
            config['num_workers'] = tld.get('num_workers')

        # A readable run name: prefer the output_dir basename (e.g.
        # 'dfine_hgnetv2_s_hrsid'), fall back to the model registry name.
        name = None
        if self.output_dir is not None:
            name = self.output_dir.name
        if not name:
            name = cfg.get('model', 'det')

        # Resolve a run id so a resumed training continues the same W&B run
        # instead of starting a new one. Only when resuming (-r): prefer the
        # id restored from the checkpoint, then fall back to the
        # wandb/latest-run symlink in this output_dir (attaches runs whose
        # checkpoints predate run-id persistence). Fresh training always
        # starts a new run.
        run_id = getattr(self, '_wandb_run_id', None)
        run_id_source = 'checkpoint' if run_id else None
        if run_id is None and getattr(self.cfg, 'resume', None) and self.output_dir is not None:
            latest = self.output_dir / 'wandb' / 'latest-run'
            try:
                if latest.exists():
                    # run dirs are named run-YYYYMMDD_HHMMSS-<id>
                    base = latest.resolve().name
                    if base.startswith('run-'):
                        parsed = base.rsplit('-', 1)[-1]
                        if parsed:
                            run_id, run_id_source = parsed, 'latest-run'
            except OSError:
                pass
        #if run_id is None:
        #    run_id = wandb.util.generate_id()
        if run_id is None:
            # wandb.util.generate_id was removed in wandb>=0.29
            gen = getattr(wandb.util, 'generate_id', None)
            if gen is None:
                try:
                    from wandb.sdk.lib import runid
                    gen = runid.generate_id
                except ImportError:
                    import uuid
                    gen = lambda: uuid.uuid4().hex[:8]
            run_id = gen()
        try:
            wandb.init(
                entity=self._WANDB_ENTITY,
                project=self._WANDB_PROJECT,
                name=name,
                dir=str(self.output_dir) if self.output_dir else None,
                config=config,
                reinit="finish_previous",
                id=run_id,
                resume="allow",
            )
            self._wandb = wandb
            # Authoritative id (wandb normalizes whatever we passed).
            self._wandb_run_id = wandb.run.id
            if run_id_source:
                print(f"[wandb] resuming run {self._wandb_run_id} (id from {run_id_source})")
            else:
                print(f"[wandb] new run {self._wandb_run_id}")
        except Exception as e:
            # Network/auth failures must not abort training.
            print(f"[wandb] init failed ({type(e).__name__}: {e}); skipping W&B logging.")
            self._wandb = None

    def _wandb_log(self, train_stats, apar_stats, epoch, epoch_time):
        """Push one per-epoch record to W&B (no-op if disabled)."""
        if self._wandb is None:
            return
        try:
            payload = {}
            if train_stats:
                for k, v in train_stats.items():
                    payload[f"train/{k}"] = v
            if apar_stats is not None:
                names = self._APAR_HEADER[1:]  # drop leading 'epoch'
                for name, v in zip(names, apar_stats):
                    payload[f"test/{name}"] = v
            if epoch_time is not None:
                payload["epoch_time_s"] = epoch_time
            payload["epoch"] = epoch
            self._wandb.log(payload, step=epoch)
        except Exception as e:
            print(f"[wandb] log failed at epoch {epoch} "
                  f"({type(e).__name__}: {e}); continuing.")

    def _wandb_finish(self):
        """Close the W&B run if one was started."""
        if self._wandb is None:
            return
        try:
            self._wandb.finish()
        except Exception:
            pass
        self._wandb = None