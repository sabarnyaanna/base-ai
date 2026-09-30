import base64
import io
import dash
from dash import dcc, html, Input, Output, State
import numpy as np
from PIL import Image

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Лабораторна робота №2"

weights = np.zeros(25)

def parse_image(contents):
    if contents is None:
        return None
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded))
    return image


def extract_features(img_pil):
    img_gray = img_pil.convert("RGB").convert("L")
    img_np = np.array(img_gray)
    h, w = img_np.shape

    cell_h = h / 5
    cell_w = w / 5
    abs_vector = []

    for r in range(5):
        for c in range(5):
            r_start = int(r * cell_h)
            r_end = int((r + 1) * cell_h)
            c_start = int(c * cell_w)
            c_end = int((c + 1) * cell_w)

            cell = img_np[r_start:r_end, c_start:c_end]
            count = 0
            for row in cell:
                for pixel in row:
                    if pixel < 128:
                        count += 1
            abs_vector.append(count)

    abs_vector = np.array(abs_vector, dtype=float)
    total_sum = np.sum(abs_vector)

    if total_sum > 0:
        norm_vector = abs_vector / total_sum
    else:
        norm_vector = abs_vector.copy()

    mean_val = np.mean(abs_vector)
    bipolar_vector = []
    for val in abs_vector:
        if val > mean_val:
            bipolar_vector.append(1)
        else:
            bipolar_vector.append(-1)

    return (
        abs_vector,
        norm_vector,
        np.array(bipolar_vector),
        img_np,
        h,
        w,
    )


card_style = {
    "backgroundColor": "#ffffff",
    "padding": "20px",
    "borderRadius": "12px",
    "boxShadow": "0 4px 6px rgba(0,0,0,0.05)",
    "marginBottom": "20px",
    "border": "1px solid #e2e8f0",
}

upload_btn_style = {
    "border": "2px dashed #3b82f6",
    "borderRadius": "8px",
    "padding": "15px",
    "textAlign": "center",
    "backgroundColor": "#eff6ff",
    "cursor": "pointer",
    "color": "#1e40af",
    "fontWeight": "bold",
}

app.layout = html.Div(
    style={
        "backgroundColor": "#f8fafc",
        "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
        "minHeight": "100vh",
        "padding": "30px",
    },
    children=[
        html.H1(
            "Лабораторна робота №2: Дослідження Нейромереж",
            style={
                "textAlign": "center",
                "color": "#1e293b",
                "marginBottom": "30px",
            },
        ),
        dcc.Tabs(
            id="tabs",
            value="tab-1",
            colors={
                "border": "#cbd5e1",
                "primary": "#3b82f6",
                "background": "#f1f5f9",
            },
            children=[
                dcc.Tab(
                    label="Завдання 1: Перцептрон",
                    value="tab-1",
                    style={"padding": "12px", "fontWeight": "bold"},
                    selected_style={"padding": "12px", "fontWeight": "bold"},
                ),
                dcc.Tab(
                    label="Завдання 2: Мережа Хопфілда",
                    value="tab-2",
                    style={"padding": "12px", "fontWeight": "bold"},
                    selected_style={"padding": "12px", "fontWeight": "bold"},
                ),
                dcc.Tab(
                    label="Завдання 3: Мережа Хеммінга",
                    value="tab-3",
                    style={"padding": "12px", "fontWeight": "bold"},
                    selected_style={"padding": "12px", "fontWeight": "bold"},
                ),
            ],
        ),
        html.Div(
            id="tab-content", style={"marginTop": "20px", "maxWidth": "900px", "margin": "20px auto"}
        ),
    ],
)


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "tab-1":
        return html.Div([
            html.Div(style=card_style, children=[
                html.H3("1. Налаштування Перцептрона", style={"color": "#0f172a", "marginTop": "0"}),
                html.Div(style={"display": "flex", "gap": "20px", "marginBottom": "15px"}, children=[
                    html.Div([
                        html.Label("Швидкість навчання (r):", style={"fontWeight": "600", "display": "block", "marginBottom": "5px"}),
                        dcc.Input(id="lr-input", type="number", value=0.1, style={"padding": "8px", "borderRadius": "6px", "border": "1px solid #cbd5e1", "width": "100%"}),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Label("Поріг (b):", style={"fontWeight": "600", "display": "block", "marginBottom": "5px"}),
                        dcc.Input(id="bias-input", type="number", value=0.0, style={"padding": "8px", "borderRadius": "6px", "border": "1px solid #cbd5e1", "width": "100%"}),
                    ], style={"flex": "1"}),
                ]),
            ]),
            
            html.Div(style=card_style, children=[
                html.H3("2. Завантаження малюнків та визначення класів", style={"color": "#0f172a", "marginTop": "0"}),
                html.Div(style={"display": "flex", "gap": "20px", "marginBottom": "20px"}, children=[
                    html.Div(style={"flex": "1"}, children=[
                        html.Label("Зображення 1:", style={"fontWeight": "bold", "display": "block", "marginBottom": "5px"}),
                        dcc.Upload(id="u-perceptron-1", children=html.Div(" Натисніть або перетягніть малюнок 1"), style=upload_btn_style),
                        html.Br(),
                        html.Label("Класифікація малюнка 1:", style={"fontWeight": "600"}),
                        dcc.RadioItems(
                            id="target-class-1",
                            options=[
                                {"label": " Клас +1 ", "value": 1},
                                {"label": " Клас -1 ", "value": -1},
                            ],
                            value=1,
                            labelStyle={"display": "block", "margin": "5px 0", "cursor": "pointer"}
                        ),
                    ]),
                    html.Div(style={"flex": "1"}, children=[
                        html.Label("Зображення 2:", style={"fontWeight": "bold", "display": "block", "marginBottom": "5px"}),
                        dcc.Upload(id="u-perceptron-2", children=html.Div(" Натисніть або перетягніть малюнок 2"), style=upload_btn_style),
                        html.Br(),
                        html.Label("Класифікація малюнка 2:", style={"fontWeight": "600"}),
                        dcc.RadioItems(
                            id="target-class-2",
                            options=[
                                {"label": " Клас +1 ", "value": 1},
                                {"label": " Клас -1 ", "value": -1},
                            ],
                            value=-1,
                            labelStyle={"display": "block", "margin": "5px 0", "cursor": "pointer"}
                        ),
                    ]),
                ]),
                html.Button(
                    "Навчити / Скоригувати ваги (на обох зображеннях)",
                    id="btn-train",
                    n_clicks=0,
                    style={
                        "backgroundColor": "#10b981",
                        "color": "white",
                        "border": "none",
                        "padding": "12px 24px",
                        "borderRadius": "8px",
                        "fontSize": "16px",
                        "fontWeight": "bold",
                        "cursor": "pointer",
                        "width": "100%"
                    }
                ),
            ]),
            html.Div(id="out-perceptron")
        ])

    elif tab == "tab-2":
        return html.Div([
            html.Div(style=card_style, children=[
                html.H3("1. Еталонні образи (Пам'ять мережі)", style={"color": "#0f172a", "marginTop": "0"}),
                html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "15px"}, children=[
                    dcc.Upload(id="u-h1", children="Завантажити Еталон 1", style=upload_btn_style),
                    dcc.Upload(id="u-h2", children="Завантажити Еталон 2", style=upload_btn_style),
                    dcc.Upload(id="u-h3", children="Завантажити Еталон 3", style=upload_btn_style),
                ])
            ]),
            html.Div(style=card_style, children=[
                html.H3("2. Тестовий образ для відновлення", style={"color": "#0f172a", "marginTop": "0"}),
                dcc.Upload(id="u-h-test", children=" Завантажити зашумлений образ", style=upload_btn_style),
            ]),
            html.Div(id="out-hopfield")
        ])

    elif tab == "tab-3":
        return html.Div([
            html.Div(style=card_style, children=[
                html.H3("1. База еталонних класів", style={"color": "#0f172a", "marginTop": "0"}),
                html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "15px"}, children=[
                    dcc.Upload(id="u-ham1", children="Завантажити Еталон 1", style=upload_btn_style),
                    dcc.Upload(id="u-ham2", children="Завантажити Еталон 2", style=upload_btn_style),
                    dcc.Upload(id="u-ham3", children="Завантажити Еталон 3", style=upload_btn_style),
                ])
            ]),
            html.Div(style=card_style, children=[
                html.H3("2. Класифікація", style={"color": "#0f172a", "marginTop": "0"}),
                dcc.Upload(id="u-ham-test", children=" Завантажити тестове зображення", style=upload_btn_style),
            ]),
            html.Div(id="out-hamming")
        ])


@app.callback(
    Output("out-perceptron", "children"),
    Input("btn-train", "n_clicks"),
    [
        State("u-perceptron-1", "contents"),
        State("target-class-1", "value"),
        State("u-perceptron-2", "contents"),
        State("target-class-2", "value"),
        State("lr-input", "value"),
        State("bias-input", "value"),
    ],
)
def train_perceptron(n_clicks, contents1, target1, contents2, target2, lr, bias):
    global weights

    if contents1 is None or contents2 is None:
        return html.Div(" Очікую завантаження ОБОХ файлів...", style={"color": "#64748b", "textAlign": "center"})

    img1 = parse_image(contents1)
    img2 = parse_image(contents2)

    _, norm_vec1, _, _, _, _ = extract_features(img1)
    _, norm_vec2, _, _, _, _ = extract_features(img2)

    samples = [
        {"vec": norm_vec1, "target": target1, "name": "Зображення 1"},
        {"vec": norm_vec2, "target": target2, "name": "Зображення 2"},
    ]

    if n_clicks > 0:
        for sample in samples:
            vec = sample["vec"]
            target = sample["target"]
            s = np.dot(weights, vec) - bias
            y = 1 if s >= 0 else -1
            error = target - y
            weights = weights + lr * error * vec

    results = []
    all_correct = True
    for sample in samples:
        vec = sample["vec"]
        target = sample["target"]
        s = np.dot(weights, vec) - bias
        y = 1 if s >= 0 else -1
        is_correct = (y == target)
        if not is_correct:
            all_correct = False
        results.append({
            "name": sample["name"],
            "pred": y,
            "target": target,
            "correct": is_correct
        })

    res_elements = []
    for r in results:
        status_color = "#10b981" if r["correct"] else "#ef4444"
        status_text = " Успіх" if r["correct"] else " Помилка"
        res_elements.append(
            html.Div(style={"marginBottom": "10px", "padding": "10px", "border": f"1px solid {status_color}", "borderRadius": "6px"}, children=[
                html.Strong(f"{r['name']}: "),
                html.Span(f"{status_text} | ", style={"color": status_color, "fontWeight": "bold"}),
                html.Span(f"Отримано Клас {r['pred']} (Очікувалось: Клас {r['target']})")
            ])
        )

    summary_color = "#10b981" if all_correct else "#f59e0b"
    summary_text = "Обидва зображення розпізнано правильно!" if all_correct else "Потрібно додаткове коригування (натисніть кнопку ще раз)."

    return html.Div(style=card_style, children=[
        html.H3("Результати обчислення:", style={"marginTop": "0"}),
        html.Div(summary_text, style={"color": summary_color, "fontWeight": "bold", "fontSize": "18px", "marginBottom": "15px"}),
        *res_elements,
        html.Hr(),
        html.P("Поточний стан ваг (пам'ять Перцептрона):", style={"fontWeight": "bold"}),
        html.Pre(str(np.round(weights, 3).tolist()), style={"backgroundColor": "#f1f5f9", "padding": "10px", "borderRadius": "6px", "overflowX": "auto"})
    ])


@app.callback(
    Output("out-hopfield", "children"),
    [
        Input("u-h1", "contents"),
        Input("u-h2", "contents"),
        Input("u-h3", "contents"),
        Input("u-h-test", "contents"),
    ],
)
def run_hopfield(h1, h2, h3, test):
    if not (h1 and h2 and h3 and test):
        return html.Div(" Будь ласка, завантажте всі 3 еталони та зашумлений образ.", style={"color": "#64748b", "textAlign": "center"})

    e1 = extract_features(parse_image(h1))[2]
    e2 = extract_features(parse_image(h2))[2]
    e3 = extract_features(parse_image(h3))[2]
    test_vec = extract_features(parse_image(test))[2]

    patterns = [e1, e2, e3]
    n = len(test_vec)
    w = np.zeros((n, n))

    for p in patterns:
        for i in range(n):
            for j in range(n):
                if i != j:
                    w[i][j] += p[i] * p[j]

    y = test_vec.copy()
    for step in range(5):
        new_y = np.dot(w, y)
        for i in range(n):
            if new_y[i] >= 0:
                y[i] = 1
            else:
                y[i] = -1

    return html.Div(style=card_style, children=[
        html.H3("Результат відновлення образу:", style={"marginTop": "0", "color": "#1e40af"}),
        html.P("Вхідний (зашумлений) вектор:", style={"fontWeight": "bold"}),
        html.Pre(str(test_vec.tolist()), style={"backgroundColor": "#f1f5f9", "padding": "10px", "borderRadius": "6px"}),
        html.P("Відновлений мережею вектор:", style={"fontWeight": "bold"}),
        html.Pre(str(y.tolist()), style={"backgroundColor": "#e0f2fe", "padding": "10px", "borderRadius": "6px"})
    ])


@app.callback(
    Output("out-hamming", "children"),
    [
        Input("u-ham1", "contents"),
        Input("u-ham2", "contents"),
        Input("u-ham3", "contents"),
        Input("u-ham-test", "contents"),
    ],
)
def run_hamming(e1, e2, e3, test):
    if not (e1 and e2 and e3 and test):
        return html.Div(" Будь ласка, завантажте всі 3 еталони та тестовий образ.", style={"color": "#64748b", "textAlign": "center"})

    p1 = extract_features(parse_image(e1))[1]
    p2 = extract_features(parse_image(e2))[1]
    p3 = extract_features(parse_image(e3))[1]
    test_vec = extract_features(parse_image(test))[1]

    patterns = [p1, p2, p3]
    n = len(test_vec)
    m = len(patterns)

    y1 = []
    for k in range(m):
        w_k = patterns[k] / 2.0
        b_k = n / 2.0
        s = np.dot(w_k, test_vec) + b_k
        y1.append(s)

    y2 = np.array(y1, dtype=float)
    eps = 0.1 / m

    for iter in range(20):
        sum_y2 = np.sum(y2)
        y2_next = np.zeros(m)
        for i in range(m):
            val = y2[i] - eps * (sum_y2 - y2[i])
            if val < 0:
                val = 0
            y2_next[i] = val

        y2 = y2_next

        active_count = 0
        for val in y2:
            if val > 0:
                active_count += 1
        if active_count <= 1:
            break

    best_class = np.argmax(y2) + 1

    return html.Div(style=card_style, children=[
        html.H3(f"🏆 Результат: Зображення належить до КЛАСУ №{best_class}", style={"color": "#047857", "marginTop": "0"}),
        html.Hr(),
        html.P(f"Виходи 1-го шару (початкова схожість): {np.round(y1, 2).tolist()}"),
        html.P(f"Виходи 2-го шару (після гальмування): {np.round(y2, 2).tolist()}"),
    ])


if __name__ == "__main__":
    app.run(debug=True, port=8051)