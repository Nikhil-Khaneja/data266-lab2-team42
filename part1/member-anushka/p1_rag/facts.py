"""Structured fact layer with the knowledge-base precedence rules from kb_manifest.json:

1. an explicit revision / erratum supersedes the earlier value,
2. an outturn supersedes a forecast for the same variable and year,
3. high/medium-trust sources that disagree with no revision note -> "conflict",
   preferring the higher-trust value,
4. low-trust content is never evidence.

The facts come from the parsed tables and stated revisions (see ingest.py); nothing is
specific to any question.
"""
TRUST_RANK = {"high": 2, "medium": 1, "low": 0}
TOL = 0.05


def fact_cites(f):
    out = []
    if f.get("label"):
        out.append(f"{f['doc_id']}:{f['label']}")
    out.append(f"{f['doc_id']}:p{f['page']}")
    return out


def fact_src(f):
    return f"{f['doc_id']} {f['label']}".strip() if f.get("label") else f"{f['doc_id']} p.{f['page']}"


def fmt(v):
    return f"{v:.1f}" if abs(round(v, 1) - v) < 1e-9 else f"{v:.2f}"


class FactStore:
    def __init__(self, facts, docs):
        self.vintage = {d["doc_id"]: (d.get("vintage_year") or 0) for d in docs}
        self.facts = [f for f in facts if f["trust"] != "low" and f["year"] and f["entity"] and f["variable"]
                      and f["value"] is not None]
        # multi-year averages ("2000-2007 (average)" columns): looked up by period, never mixed with single years
        self.period_facts = [f for f in facts if f["trust"] != "low" and not f["year"] and f.get("period")
                             and f["entity"] and f["variable"] and f["value"] is not None]
        self.max_year = max(f["year"] for f in self.facts)
        self.entities = {f["entity"] for f in self.facts}

    def lookup(self, entity, variable, year):
        return [f for f in self.facts if f["entity"] == entity and f["variable"] == variable and f["year"] == year]

    def resolve(self, entity, variable, year):
        fs = self.lookup(entity, variable, year)
        if not fs:
            return None
        key = f"{entity}, {variable}, {year}"
        revs = [f for f in fs if f.get("revised_from") is not None]
        if revs:
            r = max(revs, key=lambda f: self.vintage.get(f["doc_id"], 0))
            old = [f for f in fs if f is not r and abs(f["value"] - r["revised_from"]) < TOL]
            return {"key": key, "entity": entity, "variable": variable, "year": year, "status": "answered",
                    "value": r["value"], "best": r, "replaced": [(r["revised_from"], old[0] if old else None, "revised")],
                    "others": [],
                    "note": f"{key}: {fmt(r['value'])} after a stated revision ({fact_src(r)}), replacing "
                            f"{fmt(r['revised_from'])}" + (f" ({fact_src(old[0])})" if old else "") + "."}
        actual = [f for f in fs if not f["forecast"]]
        fcast = [f for f in fs if f["forecast"]]
        if not actual:
            best = max(fcast, key=lambda f: (TRUST_RANK[f["trust"]], self.vintage.get(f["doc_id"], 0)))
            return {"key": key, "entity": entity, "variable": variable, "year": year, "status": "answered",
                    "value": best["value"], "best": best, "replaced": [], "others": [], "forecast_only": True,
                    "note": f"{key}: {fmt(best['value'])} is a forecast ({fact_src(best)}); no outturn is reported."}
        # distinct values among actual (non-forecast) values
        groups = []
        for f in sorted(actual, key=lambda f: (-TRUST_RANK[f["trust"]], -self.vintage.get(f["doc_id"], 0))):
            for g in groups:
                if abs(g[0]["value"] - f["value"]) < TOL:
                    g.append(f)
                    break
            else:
                groups.append([f])
        best = groups[0][0]
        replaced = [(f["value"], f, "forecast") for f in fcast if abs(f["value"] - best["value"]) >= TOL]
        # keep one replaced forecast per distinct value, prefer the original (DOC1) source
        seen, rep = set(), []
        for v, f, why in sorted(replaced, key=lambda t: self.vintage.get(t[1]["doc_id"], 0)):
            if round(v, 2) not in seen:
                seen.add(round(v, 2))
                rep.append((v, f, why))
        res = {"key": key, "entity": entity, "variable": variable, "year": year, "value": best["value"],
               "best": best, "replaced": rep, "others": []}
        if len(groups) > 1:
            others = [g[0] for g in groups[1:]]
            res["status"] = "conflict"
            res["others"] = others
            res["note"] = (f"{key}: sources disagree with no revision note - "
                           + "; ".join(f"{fact_src(f)} ({f['trust']} trust) = {fmt(f['value'])}" for f in [best] + others)
                           + f". Prefer the higher-trust value {fmt(best['value'])}.")
        else:
            res["status"] = "answered"
            res["note"] = f"{key}: {fmt(best['value'])} ({fact_src(best)}"
            res["note"] += ", outturn)" if best["doc_id"] != "DOC1" or not best["forecast"] else ")"
            if rep:
                res["note"] += " - replaces the forecast " + ", ".join(f"{fmt(v)} ({fact_src(f)})" for v, f, _ in rep)
            res["note"] += "."
        return res

    def periods(self, entity, variable):
        return sorted({f["period"] for f in self.period_facts if f["entity"] == entity and f["variable"] == variable})

    def resolve_period(self, entity, variable, period):
        """Average over a period such as '2000-2007' (prefers the higher-trust source)."""
        fs = [f for f in self.period_facts if f["entity"] == entity and f["variable"] == variable
              and f["period"] == period]
        if not fs:
            return None
        best = max(fs, key=lambda f: (TRUST_RANK[f["trust"]], self.vintage.get(f["doc_id"], 0)))
        key = f"{entity}, {variable}, {period} average"
        return {"key": key, "entity": entity, "variable": variable, "year": int(period.split("-")[-1]),
                "period": period, "status": "answered", "value": best["value"], "best": best, "replaced": [],
                "others": [], "note": f"{key}: {fmt(best['value'])} ({fact_src(best)})."}
