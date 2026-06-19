// NeuroCUDA — Auto-generated testbench
module tb_neurocuda_fpga_demo;

    reg clk = 0;
    reg rst = 1;
    reg [2:0] spike_input = 0;
    wire [2:0] spike_output;
    wire [47:0] membranes;

    // Clock generation: 100 MHz
    always #5 clk = ~clk;  // 10 ns period

    // DUT instantiation
    neurocuda_fpga_demo dut (
        .clk(clk), .rst(rst),
        .spike_input(spike_input),
        .spike_output(spike_output),
        .membranes(membranes)
    );

    // Test sequence: apply Poisson spikes for T=64 timesteps
    initial begin
        $display('NeuroCUDA SNN RTL Simulation — T=64');
        #10 rst = 0;  // Release reset

        // Apply random spike input
        repeat(640) @(posedge clk) begin
            spike_input[0] <= $urandom % 2;  // Random input spikes
        end

        $display('Simulation complete. Spike output: %b', spike_output);
        $finish;
    end

endmodule
