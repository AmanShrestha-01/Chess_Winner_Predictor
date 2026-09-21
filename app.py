"""Gradio web app for the chess winner predictor.

Run it with:  .venv/bin/python app.py
Then open the URL it prints (http://127.0.0.1:7860).
"""
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent / "src"))
from chess_model import predict_winner, train_model  # noqa: E402

# Train once, when the app starts -- not on every click
print("training the model...")
MODEL, TEST_ACC = train_model()
print(f"ready. test accuracy {TEST_ACC:.1%}")


def predict(white_rating, black_rating, time_control, rated):
    """Called every time someone moves a slider. Returns what the outputs show."""
    probs = predict_winner(MODEL, int(white_rating), int(black_rating),
                           time_control, rated)

    # gr.Label wants {name: 0-to-1}, so convert back from percentages
    chart = {k: v / 100 for k, v in probs.items()}

    diff = int(white_rating) - int(black_rating)
    side = "White" if diff > 0 else "Black" if diff < 0 else "Neither side"
    summary = (
        f"**{side} is stronger by {abs(diff)} rating points.**\n\n"
        f"White wins **{probs['white']}%** · "
        f"Black wins **{probs['black']}%** · "
        f"Draw **{probs['draw']}%**\n\n"
        f"*Model: random forest, {TEST_ACC:.1%} accuracy on held-out games. "
        f"Baseline (always guess white) is 49.9%.*"
    )
    return chart, summary


with gr.Blocks(title="Chess Winner Predictor") as demo:
    gr.Markdown(
        "# Chess Winner Predictor\n"
        "Predicts the outcome from **pre-game information only** — ratings and "
        "time control. It never sees the moves."
    )

    with gr.Row():
        with gr.Column():
            white = gr.Slider(600, 2800, value=1500, step=10, label="White rating")
            black = gr.Slider(600, 2800, value=1500, step=10, label="Black rating")
            tc = gr.Dropdown(
                ["1+0", "3+0", "3+2", "5+0", "5+3", "10+0", "15+2", "30+0", "60+0"],
                value="10+0", label="Time control (minutes + increment)",
            )
            rated = gr.Checkbox(value=True, label="Rated game")
            go = gr.Button("Predict", variant="primary")

        with gr.Column():
            out_chart = gr.Label(label="Probability of each outcome", num_top_classes=3)
            out_text = gr.Markdown()

    inputs = [white, black, tc, rated]
    outputs = [out_chart, out_text]

    # Fire on button click AND whenever any input changes
    go.click(predict, inputs, outputs)
    for component in inputs:
        component.change(predict, inputs, outputs)

    demo.load(predict, inputs, outputs)   # show a result immediately on open

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
