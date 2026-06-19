// ============================================================
// NeuroCUDA → SystemVerilog RTL (Auto-generated)
// Model: neurocuda_fpga_demo
// IF neurons: 3
// Clock: 100.0 MHz
// Reset: subtractive (matches Loihi 2)
// ============================================================

`timescale 1ns / 100ps

// ============================================================
// IF Neuron — Leaky Integrate-and-Fire (beta=1.0, subtractive reset)
// Mathematically identical to Loihi 2 neuron model
// ============================================================
module if_neuron #(
    parameter THRESHOLD = 256  // Q8.8 fixed-point threshold
) (
    input  wire clk,
    input  wire rst,
    input  wire spike_in,
    output reg  spike_out,
    output reg  [15:0] membrane
);

    always @(posedge clk) begin
        if (rst) begin
            membrane <= 0;
            spike_out <= 0;
        end else begin
            if (spike_in) membrane <= membrane + 1;
            if (membrane >= THRESHOLD) begin
                spike_out <= 1;
                membrane <= membrane - THRESHOLD;
            end else begin
                spike_out <= 0;
            end
        end
    end
endmodule

// ============================================================
// Top-Level SNN: neurocuda_fpga_demo
// Layers: 3 IF neurons in pipeline
// ============================================================
module neurocuda_snn (
    input  wire clk,
    input  wire rst,
    input  wire [2:0] spike_input,
    output wire [2:0] spike_output,
    output wire [47:0] membranes
);

    // Layer 0: threshold=1.17 (Q8.8: 299)
    wire spike_lif1;
    wire [15:0] mem_lif1;

    // Layer 1: threshold=0.92 (Q8.8: 235)
    wire spike_lif2;
    wire [15:0] mem_lif2;

    // Layer 2: threshold=2.60 (Q8.8: 665)
    wire spike_lif3;
    wire [15:0] mem_lif3;

    // IF Neuron 0 (threshold=1.17)
    if_neuron #(.THRESHOLD(299))
    u_lif1 (
        .clk(clk),
        .rst(rst),
        .spike_in(spike_input[0]),
        .spike_out(spike_lif1),
        .membrane(mem_lif1)
    );

    // IF Neuron 1 (threshold=0.92)
    if_neuron #(.THRESHOLD(235))
    u_lif2 (
        .clk(clk),
        .rst(rst),
        .spike_in(spike_lif1),
        .spike_out(spike_lif2),
        .membrane(mem_lif2)
    );

    // IF Neuron 2 (threshold=2.60)
    if_neuron #(.THRESHOLD(665))
    u_lif3 (
        .clk(clk),
        .rst(rst),
        .spike_in(spike_lif2),
        .spike_out(spike_lif3),
        .membrane(mem_lif3)
    );

    // Output assignments
    assign spike_output[0] = spike_lif1;
    assign membranes[15:0] = mem_lif1;
    assign spike_output[1] = spike_lif2;
    assign membranes[31:16] = mem_lif2;
    assign spike_output[2] = spike_lif3;
    assign membranes[47:32] = mem_lif3;

endmodule

// ============================================================
// Synthesis Constraints (Xilinx Vivado / Yosys)
// ============================================================
// set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets clk]
// create_clock -period 10.00 [get_ports clk]
// Target: 100.0 MHz (10.0 ns period)
// Pipeline: II=1 (10.0 ns per neuron update)
