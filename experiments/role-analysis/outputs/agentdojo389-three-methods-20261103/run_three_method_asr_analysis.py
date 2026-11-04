"""Task-cluster Spearman bootstrap, frozen GEE results, and stratified GAMs.

GAM: binomial-logit cubic P-spline with suite effects, a second-difference
roughness penalty, weak ridge stabilization, grouped-CV smoothing selection,
and user-task bootstrap refits. There are no monotonicity constraints.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata, spearmanr

ROLES = ("system", "developer", "user", "cot", "assistant", "tool")
PARTS = {"template": "payload_without_goal", "goal": "goal"}
KNOTS = np.r_[np.zeros(4), [.2, .4, .6, .8], np.ones(4)]
LAMBDAS = np.asarray([.01, .1, 1., 10., 100., 1000.])
RIDGE = .01
LATEST_SHA = "fe602209cab66a433c34978dab8b2e8157f372ca5fb1c7f6881882401a6ffa33"


def require(value, message):
    if not value:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def csv_write(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_spearman(p, y, clusters, weights):
    """Weighted midranks reproduce a full rerank in every bootstrap sample."""
    order = np.argsort(p, kind="stable")
    ps, ys, gs = p[order], y[order], clusters[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ps) != 0) + 1]
    estimates = np.full(len(weights), np.nan)
    for begin in range(0, len(weights), 250):
        case_weights = weights[begin:begin+250, gs].astype(float)
        mass = np.add.reduceat(case_weights, starts, axis=1)
        successes = np.add.reduceat(case_weights * ys[None, :], starts, axis=1)
        midrank = np.cumsum(mass, axis=1) - mass / 2 + .5
        n = mass.sum(1)
        sx, sy = (mass * midrank).sum(1), successes.sum(1)
        sxx, sxy = (mass * midrank**2).sum(1), (successes * midrank).sum(1)
        variance = (sxx - sx*sx/n) * (sy - sy*sy/n)
        valid = variance > 1e-12
        result = np.full(len(n), np.nan)
        result[valid] = (sxy[valid] - sx[valid]*sy[valid]/n[valid]) / np.sqrt(variance[valid])
        estimates[begin:begin+len(n)] = result
    max_error = 0.
    for i in range(min(32, len(weights))):
        copies = np.repeat(np.arange(len(p)), weights[i, clusters].astype(int))
        if len(np.unique(y[copies])) < 2 or len(np.unique(p[copies])) < 2:
            require(not np.isfinite(estimates[i]), "Degenerate bootstrap rho was filled")
            continue
        direct = float(spearmanr(p[copies], y[copies]).statistic)
        error = abs(direct - estimates[i])
        require(error < 1e-10, "Weighted bootstrap rerank does not reproduce SciPy")
        max_error = max(max_error, error)
    valid = np.isfinite(estimates)
    low, high = np.quantile(estimates[valid], [.025, .975])
    return estimates, {"rho": float(spearmanr(p, y).statistic), "ci95_low": float(low), "ci95_high": float(high),
                       "valid_bootstrap_replicates": int(valid.sum()), "undefined_bootstrap_replicates": int((~valid).sum()),
                       "bootstrap_rerank_validation_max_error": max_error}


def basis(x):
    return BSpline.design_matrix(np.asarray(x), KNOTS, 3).toarray()


def design(x, suites):
    return np.column_stack([np.ones(len(x)), suites == "slack", suites == "travel", basis(x)[:, 1:]]).astype(float)


def penalty(lam):
    p = RIDGE * np.eye(10)
    differences = np.diff(np.eye(8), n=2, axis=0)[:, 1:]
    p[3:, 3:] += lam * (differences.T @ differences)
    return p


def fit_pspline(x, y, weights, lam, start=None):
    p = penalty(lam)
    beta = np.zeros(x.shape[1]) if start is None else start.copy()
    if start is None:
        beta[0] = logit((np.dot(weights, y)+.5)/(weights.sum()+1))
    # Whiten the quadratic penalty before optimization. This is an exact
    # reparameterization and avoids an ill-conditioned spline Hessian.
    chol = np.linalg.cholesky(p)
    transform = np.linalg.solve(chol.T, np.eye(len(beta)))
    z = x @ transform
    theta = chol.T @ beta
    def objective(t):
        eta = z @ t
        return float(np.dot(weights, np.logaddexp(0, eta) - y*eta) + .5*t@t)
    def gradient(t):
        return z.T @ (weights*(expit(z@t)-y)) + t
    def hessian(t):
        mu = expit(z@t)
        return z.T @ ((weights*mu*(1-mu))[:, None]*z) + np.eye(len(t))
    for iteration in range(100):
        g = gradient(theta)
        if np.max(abs(g)) < 1e-9:
            break
        step = np.linalg.solve(hessian(theta), g)
        old = objective(theta)
        factor = 1.
        while factor > 1e-12:
            candidate = theta - factor*step
            if objective(candidate) <= old - 1e-4*factor*float(g@step) + 1e-10:
                theta = candidate
                break
            factor *= .5
        else:
            break
    beta = transform @ theta
    norm = float(np.max(abs(x.T @ (weights*(expit(x@beta)-y)) + p@beta)))
    if norm >= 1e-5:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference = minimize(objective, theta, jac=gradient, hess=hessian,
                                 method="trust-exact", options={"gtol":1e-9,"maxiter":300})
        beta = transform @ reference.x
        norm = float(np.max(abs(x.T @ (weights*(expit(x@beta)-y)) + p@beta)))
    require(norm < 1e-5, f"Penalized fit did not converge: gradient={norm}, lambda={lam}, events={np.dot(weights,y)}")
    return beta, norm


def cv_lambda(x, y, folds):
    scores = []
    for lam in LAMBDAS:
        losses = []
        for fold in range(5):
            train, test = folds != fold, folds == fold
            require(train.any() and test.any(), "Empty grouped CV fold")
            beta, _ = fit_pspline(x[train], y[train], np.ones(train.sum()), lam)
            eta = x[test] @ beta
            losses.append(float(np.mean(np.logaddexp(0, eta)-y[test]*eta)))
        scores.append({"lambda": float(lam), "fold_losses": losses, "mean_loss": float(np.mean(losses)),
                       "se_loss": float(np.std(losses, ddof=1)/np.sqrt(5))})
    best = min(scores, key=lambda r: r["mean_loss"])
    selected = max(r["lambda"] for r in scores if r["mean_loss"] <= best["mean_loss"] + best["se_loss"])
    return selected, scores


def prediction_design(grid):
    matrices = np.asarray([design(grid, np.full(len(grid), suite)) for suite in ("banking", "slack", "travel")])
    derivative = np.zeros((len(grid), 10))
    derivative[:, 3:] = BSpline(KNOTS, np.eye(8), 3).derivative()(grid)[:, 1:]
    return matrices, derivative


def marginal_prediction(coefficients, grid, suite_weights):
    coefficients = np.atleast_2d(coefficients)
    matrices, derivative = prediction_design(grid)
    probabilities = expit(np.einsum("sgk,bk->bsg", matrices, coefficients))
    curve = np.einsum("s,bsg->bg", suite_weights, probabilities)
    derivative_eta = coefficients @ derivative.T
    slope = np.einsum("s,bsg->bg", suite_weights, probabilities*(1-probabilities)) * derivative_eta
    return curve, slope


def intervals(grid, mask):
    groups, start = [], None
    for i, value in enumerate(np.r_[mask, False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i-start >= 3:
                groups.append([float(grid[start]), float(grid[i-1])])
            start = None
    return groups


def validate_fit(x, y, lam, beta):
    p = penalty(lam)
    def value(b):
        eta = x@b
        return float(np.sum(np.logaddexp(0, eta)-y*eta)+.5*b@p@b)
    def gradient(b):
        return x.T@(expit(x@b)-y)+p@b
    test = np.linspace(-.2, .3, x.shape[1])
    numerical = np.empty(len(test))
    for j in range(len(test)):
        direction = np.zeros(len(test)); direction[j] = 1e-5
        numerical[j] = (value(test+direction)-value(test-direction))/(2e-5)
    gradient_error = float(np.max(abs(numerical-gradient(test))))
    require(np.allclose(numerical, gradient(test), atol=1e-6, rtol=1e-5), "Penalized objective gradient check failed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(value, beta + .1*np.sin(np.arange(len(beta))), jac=gradient,
                          method="BFGS", options={"gtol": 1e-8, "maxiter": 500})
    error = float(np.max(abs(expit(x@beta)-expit(x@result.x))))
    require(error < 1e-5, "Independent BFGS prediction check failed")
    return {"finite_difference_gradient_error": gradient_error, "independent_optimizer_probability_error": error}


def gam_worker(job):
    part, role, xvalue, y, condition, suites, clusters, folds, weights, suite_weights = job
    output, arrays, cv_records, validation = [], {}, [], []
    for cohort in ("pooled", "plain", "prerequisite"):
        mask = np.ones(len(y), bool) if cohort == "pooled" else condition == cohort
        xx, yy, gg = xvalue[mask], y[mask], clusters[mask]
        x = design(xx, suites[mask])
        require(np.max(abs(basis(xx)-BSpline(KNOTS, np.eye(8), 3)(xx))) < 1e-13, "Spline basis identity failed")
        lam, cv = cv_lambda(x, yy, folds[mask])
        cv_records.append({"part": part, "role": role, "cohort": cohort, "selected_lambda": lam, "candidates": cv})
        beta, main_gradient = fit_pspline(x, yy, np.ones(len(yy)), lam)
        validation.append({"part": part, "role": role, "cohort": cohort, **validate_fit(x, yy, lam, beta)})
        grid = np.linspace(np.quantile(xx, .01), np.quantile(xx, .99), 101)
        curve, derivative = marginal_prediction(beta, grid, suite_weights)
        curve, derivative = curve[0], derivative[0]
        endpoints = np.quantile(xx, [.1, .9])
        endpoint_curve, _ = marginal_prediction(beta, endpoints, suite_weights)
        boot_beta = np.empty((len(weights), len(beta)))
        max_gradient, zero_success_draws = main_gradient, 0
        for i, cluster_weight in enumerate(weights):
            row_weight = cluster_weight[gg].astype(float)
            require(row_weight.sum() > 0, "Empty bootstrap cohort")
            zero_success_draws += int(np.dot(row_weight, yy) == 0)
            boot_beta[i], norm = fit_pspline(x, yy, row_weight, lam, start=beta)
            max_gradient = max(max_gradient, norm)
        boot_curve, boot_derivative = marginal_prediction(boot_beta, grid, suite_weights)
        lower, upper = np.quantile(boot_curve, [.025, .975], axis=0)
        boot_endpoints, _ = marginal_prediction(boot_beta, endpoints, suite_weights)
        boot_delta = boot_endpoints[:, 1]-boot_endpoints[:, 0]
        delta_low, delta_high = np.quantile(boot_delta, [.025, .975])
        sd = np.std(boot_derivative, axis=0, ddof=1)
        active = sd > 1e-10
        max_t = np.max(abs((boot_derivative[:, active]-derivative[active])/sd[active]), axis=1)
        critical = float(np.quantile(max_t, .95))
        derivative_lower, derivative_upper = derivative-critical*sd, derivative+critical*sd
        positive = intervals(grid, derivative_lower > 0)
        negative = intervals(grid, derivative_upper < 0)
        row = {"part": part, "role": role, "cohort": cohort, "cases": len(yy), "successes": int(yy.sum()),
               "lambda": lam, "bootstrap_refits": len(weights), "zero_success_bootstrap_draws": zero_success_draws,
               "x10": float(endpoints[0]), "x90": float(endpoints[1]),
               "predicted_asr_x10": float(endpoint_curve[0,0]), "predicted_asr_x90": float(endpoint_curve[0,1]),
               "delta_x90_minus_x10": float(endpoint_curve[0,1]-endpoint_curve[0,0]),
               "delta_ci95_low": float(delta_low), "delta_ci95_high": float(delta_high),
               "derivative_positive_intervals": positive, "derivative_negative_intervals": negative,
               "positive_derivative_grid_fraction": float((derivative > 0).mean()),
               "maximum_fit_gradient": max_gradient, "simultaneous_derivative_critical_value": critical}
        output.append(row)
        for name, value in (("coefficients", beta), ("bootstrap_coefficients", boot_beta), ("grid", grid),
                            ("curve", curve), ("ci95_low", lower), ("ci95_high", upper), ("derivative", derivative),
                            ("derivative_sim95_low", derivative_lower), ("derivative_sim95_high", derivative_upper)):
            arrays[cohort + "_" + name] = value
    return part, role, output, arrays, cv_records, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gam-bootstrap", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root, source = args.output.resolve(), args.predictions.resolve()
    require(not root.exists(), "Use a new output directory")
    require(read(source/"summary.json")["probe_sha256"] == LATEST_SHA, "Wrong latest probe")
    expected = dict(line.split("  ",1)[::-1] for line in (source/"SHA256SUMS").read_text().splitlines())
    require(sha(source/"per_case_probabilities.csv") == expected["per_case_probabilities.csv"], "Probability source changed")
    previous = read(args.previous/"results.json")
    require(previous["validation"]["status"] == "passed" and previous["protocol"]["input_csv_sha256"] == sha(source/"per_case_probabilities.csv"), "Earlier validated GEE source differs")
    d = pd.read_csv(source/"per_case_probabilities.csv", keep_default_na=False, float_precision="round_trip")
    reference = d[(d.part == "goal") & (d.attack_success.astype(str) != "")].copy().reset_index(drop=True)
    require(len(reference) == 752, "Wrong scored cohort")
    ids = reference.case_id.tolist()
    groups = sorted({"/".join(key.split("/")[1:3]) for key in ids})
    require(len(groups) == 56, "Wrong user-task clusters")
    group_map = {key:i for i,key in enumerate(groups)}
    clusters = np.asarray([group_map["/".join(key.split("/")[1:3])] for key in ids])
    y = (reference.attack_success.astype(str) == "True").to_numpy(int)
    condition, suites = reference.condition.to_numpy(), reference.suite.to_numpy()
    weights = np.random.default_rng(20261103).multinomial(56, np.full(56, 1/56), size=10000).astype(np.uint16)
    require(1000 <= args.gam_bootstrap <= len(weights), "Need 1,000-10,000 GAM bootstrap draws")
    fold_group = np.zeros(56, int)
    rng = np.random.default_rng(20261103)
    for suite in ("banking", "slack", "travel"):
        eligible = [i for i,g in enumerate(groups) if g.startswith(suite+"/")]
        for j,i in enumerate(rng.permutation(eligible)):
            fold_group[i] = j % 5
    folds = fold_group[clusters]
    suite_weights = np.asarray([(suites == suite).mean() for suite in ("banking","slack","travel")])
    root.mkdir(parents=True)
    (root/"models").mkdir()
    np.savez_compressed(root/"user_task_bootstrap_weights.npz", groups=np.asarray(groups), weights=weights,
                        folds=fold_group, seed=np.asarray(20261103))
    descriptions, spearman_arrays, jobs = [], {}, []
    for part, stored in PARTS.items():
        part_data = d[(d.part == stored) & (d.attack_success.astype(str) != "")].set_index("case_id").loc[ids]
        for role in ROLES:
            p = part_data["p_"+role].to_numpy(float)
            rank = (rankdata(p, method="average")-.5)/len(p)
            for cohort in ("pooled","plain","prerequisite"):
                mask = np.ones(len(y), bool) if cohort == "pooled" else condition == cohort
                estimates, values = bootstrap_spearman(p[mask], y[mask], clusters[mask], weights)
                old = next(row for row in previous["descriptive_results"] if row["part"]==part and row["role"]==role and row["condition"]==cohort)
                require(abs(values["rho"]-old["spearman_rho"]) < 1e-12, "Earlier Spearman point estimate differs")
                descriptions.append({"part":part,"role":role,"cohort":cohort,"cases":int(mask.sum()),
                                     "successes":int(y[mask].sum()), **values})
                spearman_arrays[f"{part}_{role}_{cohort}"] = estimates
            jobs.append((part,role,rank,y,condition,suites,clusters,folds,weights[:args.gam_bootstrap],suite_weights))
            print(json.dumps({"stage":"spearman_bootstrapped","part":part,"role":role}),flush=True)
    csv_write(root/"spearman.csv",descriptions)
    np.savez_compressed(root/"spearman_bootstrap_estimates.npz",**spearman_arrays)
    csv_write(root/"gee_adjusted.csv",previous["primary_results"])
    csv_write(root/"gee_unadjusted.csv",previous["unadjusted_results"])
    gam, cv, validation = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(gam_worker,job) for job in jobs]
        for future in as_completed(futures):
            part,role,rows,arrays,records,checks = future.result()
            np.savez_compressed(root/"models"/f"{part}_{role}.npz",**arrays)
            gam.extend(rows);cv.extend(records);validation.extend(checks)
            print(json.dumps({"stage":"gam_bootstrapped","part":part,"role":role,
                              "cohorts":3,"refits_per_cohort":args.gam_bootstrap}),flush=True)
    gam.sort(key=lambda r:(r["part"],ROLES.index(r["role"]),r["cohort"]))
    csv_write(root/"gam_summary.csv",[{k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()}for row in gam])
    write(root/"gam_cv.json",cv)
    protocol = {"probe_sha256":LATEST_SHA,"cases":752,"user_task_clusters":56,"bootstrap_seed":20261103,
                "spearman_bootstrap_replicates":10000,"gam_bootstrap_replicates":args.gam_bootstrap,
                "gam":"Binomial-logit P-spline with suite fixed effects; no monotonicity constraint",
                "basis":"Cubic B-spline, 8 basis functions, internal knots at pooled probability-rank .2,.4,.6,.8; first coefficient anchored at zero",
                "penalty":"Second differences of spline coefficients plus weak ridge .01 on all coefficients",
                "smoothing":"5-fold user-task grouped CV; largest lambda within one SE of minimum mean fold log loss",
                "lambda_candidates":LAMBDAS.tolist(),"bootstrap_smoothing":"Lambda and original percentile transformation fixed during bootstrap; uncertainty is conditional on chosen smoothing",
                "suite_standardization":{"banking":float(suite_weights[0]),"slack":float(suite_weights[1]),"travel":float(suite_weights[2])},
                "curve_range":"Within-cohort 1st-99th percentiles of pooled midrank; endpoint contrast uses 10th and 90th percentiles",
                "curve_intervals":"95% pointwise user-task percentile bootstrap; derivative uses per-curve approximate max-t simultaneous 95% band, conditional on lambda",
                "multiplicity":"GEE: BH across 12 primary slopes. Spearman/GAM intervals exploratory; not family-wise adjusted across roles or conditions",
                "zero_event_draws":"Undefined Spearman correlations omitted with counts reported. GAM includes zero-success bootstrap draws through weak regularization",
                "x_axis":"Probability percentile in all 752 scored cases, identical probabilities have identical average ranks",
                "predictions_root":str(source),"previous_gee_root":str(args.previous.resolve()),
                "input_sha256":sha(source/"per_case_probabilities.csv"),"script_sha256":sha(Path(__file__))}
    results = {"protocol":protocol,"spearman":descriptions,"gee":previous["primary_results"],"gam":gam,
               "validation":{"status":"passed","spearman_scipy_checks":36*32,
                             "maximum_spearman_bootstrap_error":max(r["bootstrap_rerank_validation_max_error"]for r in descriptions),
                             "gam_fits_independently_checked":validation,"previous_gee_validation":previous["validation"]}}
    write(root/"results.json",results)
    shutil.copy2(Path(__file__),root/Path(__file__).name)
    print(json.dumps({"stage":"complete","spearman_rows":len(descriptions),"gam_rows":len(gam),"gee_rows":12}),flush=True)


if __name__ == "__main__":
    main()
