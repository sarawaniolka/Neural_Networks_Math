from manim import *
import json
import numpy as np

# Load the JSON data
with open("visualization_data.json", "r") as f:
    data = json.load(f)

layer_sizes = data["layer_sizes"]


class NeuralNetworkBase(Scene):
    def add_title(self, title_text):
        """Add a title at the top of the scene."""
        title = Text(title_text).to_edge(UP)
        self.play(Write(title))
        self.wait(0.3)

    def create_layer(self, x_position, num_nodes, color):
        """Create a layer of neurons with a specific color."""
        return VGroup(
            *[
                Dot(radius=0.1, color=color).shift(UP * i * 0.8 + RIGHT * x_position)
                for i in range(-num_nodes // 2, num_nodes // 2)
            ]
        )

    def shift_layer_and_text(self, layer, text_group, shift_distance):
        """Shift both neurons and text with a slight offset for text."""
        self.play(
            layer.animate.shift(shift_distance),
            text_group.animate.shift(shift_distance + LEFT),
        )
        return text_group

    def connect_layers(self, layer1, layer2):
        """Create connections between two layers."""
        lines = VGroup(
            *[
                Line(node1.get_center(), node2.get_center(), stroke_width=2)
                for node1 in layer1
                for node2 in layer2
            ]
        )
        self.play(FadeIn(lines))


class ArchitectureVisualization(NeuralNetworkBase):
    def construct(self):
        self.add_title("Neural Network Architecture")
        x_position = -4
        colors = [BLUE, GREEN, ORANGE, RED]
        layers = []

        for idx, size in enumerate(layer_sizes):
            layer_text = Text(f"Layer {idx + 1}: {size} neurons", font_size=24)
            layer_text.to_corner(DL)
            self.play(Write(layer_text), run_time=0.2)

            layer = self.create_layer(x_position, size, colors[idx % len(colors)])
            self.play(FadeIn(layer))
            layers.append(layer)

            self.play(FadeOut(layer_text))

            if idx > 0:
                self.connect_layers(layers[idx - 1], layers[idx])

            x_position += 3

        self.wait(2)


class ForwardPassVisualization(NeuralNetworkBase):
    def construct(self):
        self.add_title("Neural Network Forward Pass")
        input_data = data["inputs"]
        weights = data["weights"]
        biases = data["biases"]
        colors = [BLUE, GREEN, ORANGE, RED]

        x_position = -4
        layers = []

        # Create input and first hidden layer
        for idx, size in enumerate(layer_sizes[:2]):
            layer = self.create_layer(x_position, size, colors[idx % len(colors)])
            layers.append(layer)
            self.play(FadeIn(layer))
            x_position += 3

        input_texts = VGroup(
            *[
                Text(f"{value:.2f}", font_size=20).next_to(neuron, LEFT)
                for neuron, value in zip(layers[0], input_data)
            ]
        )
        self.play(FadeIn(input_texts), run_time=0.5)

        # Forward pass for the first hidden layer
        relu_outputs, text_outputs = self.animate_forward_pass(
            input_texts,
            layers[0],
            layers[1],
            weights[0],
            biases[0],
            input_data,
            colors,
            activation="relu",
        )

        # Add the second hidden layer
        second_hidden_layer = self.create_layer(
            x_position - 3, layer_sizes[2], colors[2]
        )
        layers.append(second_hidden_layer)
        self.play(FadeIn(second_hidden_layer))

        # Forward pass for the second hidden layer
        next_output, text_outputs = self.animate_forward_pass(
            text_outputs,
            layers[1],
            layers[2],
            weights[1],
            biases[1],
            relu_outputs,
            colors,
            activation="relu",
        )

        # Add the output layer
        output_layer = self.create_layer(x_position - 3, layer_sizes[3], colors[3])
        layers.append(output_layer)
        self.play(FadeIn(output_layer))

        final_output, text_outputs = self.animate_forward_pass(
            text_outputs,
            layers[2],
            layers[3],
            weights[2],
            biases[2],
            next_output,
            colors,
            activation="softmax",
        )
        self.wait(5)

    def animate_forward_pass(
        self,
        text_outputs,
        input_layer,
        hidden_layer,
        weights,
        biases,
        input_data,
        colors,
        activation,
    ):
        """Animate the forward pass for a single layer."""

        last_neuron = hidden_layer[0]
        self.play(Indicate(last_neuron, color=YELLOW), run_time=1)
        self.wait(0.5)

        formula_text = (
            Text("\u03A3(weight * input) + bias", font_size=20, color=WHITE)
            .to_edge(RIGHT)
            .shift(UP * 1.5, LEFT)
        )
        self.play(FadeIn(formula_text), run_time=0.5)

        calculation_details = VGroup()
        calculation_lines = VGroup()

        for idx, (weight, input_value, input_neuron) in enumerate(
            zip(weights[0], input_data, input_layer)
        ):
            line = Line(
                input_neuron.get_center(),
                last_neuron.get_center(),
                color=colors[idx % len(colors)],
                stroke_width=3,
            )
            calculation_lines.add(line)
            self.play(Create(line), run_time=0.3)

            calculation = Text(
                f"{weight:.2f} * {input_value:.2f} = {weight * input_value:.2f}",
                font_size=20,
                color=colors[idx % len(colors)],
            ).next_to(formula_text, DOWN * (idx + 1), aligned_edge=LEFT)
            calculation_details.add(calculation)
            self.play(FadeIn(calculation), run_time=0.2)

        bias_text = Text(
            f"+ bias = {biases[0]:.2f}", font_size=20, color=PURPLE
        ).next_to(calculation_details, DOWN, aligned_edge=LEFT)
        self.play(FadeIn(bias_text), run_time=0.5)

        weighted_sum = sum(w * x for w, x in zip(weights[0], input_data)) + biases[0]
        result_text = Text(f"= {weighted_sum:.2f}", font_size=20, color=WHITE).next_to(
            bias_text, DOWN, aligned_edge=LEFT
        )
        self.play(FadeIn(result_text), run_time=0.5)

        white_lines = VGroup()
        for i, neuron in enumerate(hidden_layer[1:]):
            for input_neuron in input_layer:
                line = Line(
                    input_neuron.get_center(),
                    neuron.get_center(),
                    color=WHITE,
                    stroke_width=2,
                )
                white_lines.add(line)
        self.play(Create(white_lines))

        activation_outputs = VGroup()
        activation_inputs = np.dot(np.array(weights), np.array(input_data)) + np.array(
            biases
        )
        numeric_activation_outputs = []

        # Define the activation function based on the type
        if activation == "relu":
            formula = Text("ReLU(x) = max(0, x)", font_size=20, color=WHITE).next_to(
                result_text, DOWN, aligned_edge=LEFT
            )
            activation_func = lambda x: max(0, x)
        elif activation == "softmax":
            formula = Text(
                "S(y_i) = exp(y_i) / sum(exp(y_j))", font_size=20, color=WHITE
            ).next_to(result_text, DOWN, aligned_edge=LEFT)
            activation_func = lambda x: np.exp(x - np.max(activation_inputs)) / np.sum(
                np.exp(x - np.max(activation_inputs))
            )
        else:
            raise ValueError(f"Unknown activation function: {activation}")

        self.play(FadeIn(formula), run_time=0.5)

        # Apply activation function and create outputs
        for hidden_neuron, output_value in zip(hidden_layer, activation_inputs):
            activated_value = activation_func(output_value)
            text = Text(f"{activated_value:.2f}", font_size=20).next_to(
                hidden_neuron, RIGHT
            )
            numeric_activation_outputs.append(activated_value)
            activation_outputs.add(text)

        self.add(activation_outputs)
        self.play(FadeIn(activation_outputs), run_time=0.5)

        self.play(
            FadeOut(
                text_outputs,
                input_layer,
                calculation_lines,
                calculation_details,
                bias_text,
                result_text,
                white_lines,
                formula,
                formula_text,
            )
        )

        output_text = self.shift_layer_and_text(
            hidden_layer, activation_outputs, LEFT * 3.5
        )
        return numeric_activation_outputs, activation_outputs
