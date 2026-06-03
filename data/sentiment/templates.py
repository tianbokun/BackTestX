TEMPLATES = {
    "minimal": {
        "name": "📋 极简白",
        "layout": "column",
        "figsize": (14, 24),
        "height_ratios": [0.07, 0.25, 0.16, 0.42, 0.10],
        "hspace": 0.65,
        "colors": {
            "bg": "#ffffff",
            "title": "#1f2937",
            "subtitle": "#6b7280",
            "pos": "#22c55e",
            "neg": "#ef4444",
            "hist": "#3b82f6",
            "grid": "lightgray",
            "table_header_bg": "#1f2937",
            "table_header_text": "#ffffff",
            "table_pos_bg": "#f0fdf4",
            "table_neg_bg": "#fef2f2",
            "table_border": "#e5e7eb",
            "footer": "#9ca3af",
            "annotation": "#1f2937",
        },
        "font_sizes": {"title": 18, "subtitle": 9, "axis": 10, "annotation": 7, "table": 9, "footer": 7.5},
        "table_n": 20,
        "show_histogram": True,
        "show_kpi_cards": False,
        "show_subtitle": True,
        "show_annotations": True,
        "annotations_n": 5,
        "decorations": None,
    },
    "dark": {
        "name": "🌙 深色科技",
        "layout": "column",
        "figsize": (14, 24),
        "height_ratios": [0.09, 0.25, 0.16, 0.38, 0.12],
        "hspace": 0.65,
        "colors": {
            "bg": "#0f172a",
            "title": "#e2e8f0",
            "subtitle": "#94a3b8",
            "pos": "#f43f5e",
            "neg": "#10b981",
            "hist": "#60a5fa",
            "grid": "#334155",
            "table_header_bg": "#1e293b",
            "table_header_text": "#e2e8f0",
            "table_pos_bg": "#7f1d1d",
            "table_neg_bg": "#064e3b",
            "table_border": "#334155",
            "footer": "#64748b",
            "annotation": "#cbd5e1",
        },
        "font_sizes": {"title": 20, "subtitle": 9, "axis": 10, "annotation": 7, "table": 8, "footer": 7},
        "table_n": 20,
        "show_histogram": True,
        "show_kpi_cards": True,
        "show_subtitle": True,
        "show_annotations": True,
        "annotations_n": 5,
        "decorations": None,
    },
    "warm": {
        "name": "🏮 暖色财经",
        "layout": "column",
        "figsize": (14, 24),
        "height_ratios": [0.08, 0.25, 0.16, 0.40, 0.11],
        "hspace": 0.65,
        "colors": {
            "bg": "#fefcf5",
            "title": "#78350f",
            "subtitle": "#92400e",
            "pos": "#dc2626",
            "neg": "#166534",
            "hist": "#b45309",
            "grid": "#d6d3d1",
            "table_header_bg": "#78350f",
            "table_header_text": "#fefcf5",
            "table_pos_bg": "#fef2f2",
            "table_neg_bg": "#f0fdf4",
            "table_border": "#d6d3d1",
            "footer": "#a8a29e",
            "annotation": "#44403c",
        },
        "font_sizes": {"title": 18, "subtitle": 8, "axis": 10, "annotation": 7, "table": 9, "footer": 7},
        "table_n": 20,
        "show_histogram": True,
        "show_kpi_cards": False,
        "show_subtitle": True,
        "show_annotations": True,
        "annotations_n": 5,
        "decorations": "double_line",
    },
    "xiaohongshu": {
        "name": "📱 小红书风",
        "layout": "column_compact",
        "figsize": (9, 16),
        "height_ratios": [0.10, 0.32, 0.42, 0.16],
        "hspace": 0.55,
        "colors": {
            "bg": "#fffaf5",
            "title": "#1c1917",
            "subtitle": "#78716c",
            "pos": "#e11d48",
            "neg": "#059669",
            "hist": "#f43f5e",
            "grid": "#e7e5e4",
            "table_header_bg": "#1c1917",
            "table_header_text": "#fffaf5",
            "table_pos_bg": "#fff1f2",
            "table_neg_bg": "#f0fdf4",
            "table_border": "#e7e5e4",
            "footer": "#a8a29e",
            "annotation": "#292524",
        },
        "font_sizes": {"title": 22, "subtitle": 10, "axis": 10, "annotation": 8, "table": 10, "footer": 7},
        "table_n": 5,
        "show_histogram": False,
        "show_kpi_cards": False,
        "show_subtitle": True,
        "show_annotations": True,
        "annotations_n": 5,
        "decorations": None,
    },
    "dashboard": {
        "name": "📊 专业仪表盘",
        "layout": "grid",
        "figsize": (20, 11),
        "height_ratios": [0.08, 0.44, 0.44, 0.10],
        "hspace": 0.50,
        "colors": {
            "bg": "#ffffff",
            "title": "#1e3a5f",
            "subtitle": "#4a5568",
            "pos": "#dc2626",
            "neg": "#16a34a",
            "hist": "#2563eb",
            "grid": "#cbd5e1",
            "table_header_bg": "#1e3a5f",
            "table_header_text": "#ffffff",
            "table_pos_bg": "#fef2f2",
            "table_neg_bg": "#f0fdf4",
            "table_border": "#cbd5e1",
            "footer": "#64748b",
            "annotation": "#334155",
        },
        "font_sizes": {"title": 20, "subtitle": 10, "axis": 10, "annotation": 7, "table": 8, "footer": 8},
        "table_n": 10,
        "show_histogram": True,
        "show_kpi_cards": True,
        "show_subtitle": True,
        "show_annotations": True,
        "annotations_n": 5,
        "decorations": "header_bar",
    },
}

TEMPLATE_ORDER = ["minimal", "dark", "warm", "xiaohongshu", "dashboard"]


def resolve_template(name: str, overrides: dict, preview: bool = False) -> dict:
    import copy
    cfg = copy.deepcopy(TEMPLATES.get(name, TEMPLATES["minimal"]))

    for k in cfg["colors"]:
        if k in overrides:
            cfg["colors"][k] = overrides[k]

    fs = cfg.get("font_sizes", {})
    for k in list(fs.keys()):
        if k in overrides:
            fs[k] = overrides[k]

    for k in ["table_n", "show_histogram", "show_kpi_cards", "show_subtitle", "show_annotations"]:
        if k in overrides:
            cfg[k] = overrides[k]

    if "figsize_width" in overrides:
        w = overrides["figsize_width"]
        h = overrides.get("figsize_height", cfg["figsize"][1])
        cfg["figsize"] = (w, h)
    elif "figsize_height" in overrides:
        cfg["figsize"] = (cfg["figsize"][0], overrides["figsize_height"])
    if "figsize_width" not in overrides and "figsize_height" not in overrides:
        w_scale = overrides.get("figsize_scale_w", 1.0)
        h_scale = overrides.get("figsize_scale_h", 1.0)
        if w_scale != 1.0 or h_scale != 1.0:
            cfg["figsize"] = (cfg["figsize"][0] * w_scale, cfg["figsize"][1] * h_scale)

    for k in ["hspace", "annotations_n"]:
        if k in overrides:
            cfg[k] = overrides[k]

    for lk in ["height_ratios"]:
        if lk in overrides:
            cfg[lk] = overrides[lk]

    if preview:
        factor = overrides.get("preview_scale", 0.5)
        cfg["figsize"] = (cfg["figsize"][0] * factor, cfg["figsize"][1] * factor)
        for k in cfg.get("font_sizes", {}):
            cfg["font_sizes"][k] = max(4, int(cfg["font_sizes"].get(k, 9) * factor * 0.7))
        cfg["_dpi"] = 72
    else:
        cfg["_dpi"] = 200

    if not cfg.get("show_histogram", True) and cfg.get("layout") != "column_compact":
        pass

    return cfg


def template_options():
    return [(k, v["name"]) for k, v in TEMPLATES.items()]


def template_default_overrides(name: str) -> dict:
    tmpl = TEMPLATES.get(name, TEMPLATES["minimal"])
    result = {}
    for k, v in tmpl["colors"].items():
        result[k] = v
    result["table_n"] = tmpl["table_n"]
    result.update(tmpl["font_sizes"])
    result["show_histogram"] = tmpl["show_histogram"]
    result["show_kpi_cards"] = tmpl["show_kpi_cards"]
    result["show_subtitle"] = tmpl["show_subtitle"]
    result["show_annotations"] = tmpl["show_annotations"]
    result["figsize_width"] = tmpl["figsize"][0]
    result["figsize_height"] = tmpl["figsize"][1]
    result["hspace"] = tmpl["hspace"]
    result["annotations_n"] = tmpl["annotations_n"]
    return result
