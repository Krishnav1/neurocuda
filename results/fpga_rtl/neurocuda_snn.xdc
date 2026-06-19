create_clock -period 10.000 -name clk [get_ports clk]
set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets clk]
