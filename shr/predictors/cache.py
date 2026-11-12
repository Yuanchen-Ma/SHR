"""Resume individual ensemble fits."""

import inspect
import time

from shr.io import fingerprint, now, read, write


def cached_trainer(output, original, signature):
    import torch
    from shr.predictors import training
    from shr.predictors.registry import count_trainable_parameters, make_rul_model

    def run(*args, **kwargs):
        bound = inspect.signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        label = arguments.get("progress_label") or str(arguments["seed"])
        identity = dict(signature=signature, config=arguments["config"],
                        seed=arguments["seed"], label=label,
                        function=original.__name__, epochs=arguments.get("epochs"))
        folder = output / "fit_cache" / fingerprint(identity)[:24]
        marker = folder / "complete.json"
        if marker.exists():
            metadata = read(marker)
            if metadata["identity"] != identity:
                raise ValueError("Training cache identity mismatch")
            results = []
            files = sorted(folder.glob("member_*.pt"))
            if len(files) != arguments["config"]["model"]["ensemble_size"]:
                raise ValueError("Incomplete ensemble cache")
            for index, path in enumerate(files):
                state = torch.load(path, map_location="cpu", weights_only=False)
                model = make_rul_model(state["input_dim"], arguments["config"])
                model.load_state_dict(state["state_dict"])
                model.to(arguments["device"])
                results.append(training.TrainingResult(
                    model, state["best_epoch"], state["best_validation_loss"],
                    metadata["history"][index],
                ))
            return results
        print("TRAIN", label, flush=True)
        start = time.perf_counter()
        results = original(*args, **kwargs)
        training.save_ensemble(results, folder, {
            "architecture": arguments["config"]["model"]["architecture"],
            "extension_signature": signature,
        })
        write(marker, dict(identity=identity, seconds=time.perf_counter() - start,
                           trainable_parameters=count_trainable_parameters(results[0].model),
                           history=[result.history for result in results]))
        return results

    return run
