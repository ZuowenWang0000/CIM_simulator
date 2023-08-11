
    $display("execute layer_0");
    layer_0.execute(
            .clk                    (clk),
            .data_bus               (data_bus),
            .weight_bus             (weight_bus),
            .control_bus            (control_bus),
            .next_command_interrupt (next_command_interrupt),
            .output_line_stored     (output_line_stored)
        );
    layer_0.validate_outputs();

        
    $display("execute layer_1");
    layer_1.row_i = layer_0.output_row_i;
    // layer_1.row_i = layer_0.ground_truth_output_row_i;
    layer_1.row_i_number_of_entries = layer_0.output_row_i_number_of_entries;
    layer_1.execute(
            .clk                    (clk),
            .data_bus               (data_bus),
            .weight_bus             (weight_bus),
            .control_bus            (control_bus),
            .next_command_interrupt (next_command_interrupt),
            .output_line_stored     (output_line_stored)
        );
    layer_1.validate_outputs();

        
    $display("execute layer_2");
    layer_2.row_i = layer_1.output_row_i;
    // layer_2.row_i = layer_1.ground_truth_output_row_i;
    layer_2.row_i_number_of_entries = layer_1.output_row_i_number_of_entries;
    layer_2.execute(
            .clk                    (clk),
            .data_bus               (data_bus),
            .weight_bus             (weight_bus),
            .control_bus            (control_bus),
            .next_command_interrupt (next_command_interrupt),
            .output_line_stored     (output_line_stored)
        );
    layer_2.validate_outputs();

        
    $display("execute layer_3");
    layer_3.row_i = layer_2.output_row_i;
    // layer_3.row_i = layer_2.ground_truth_output_row_i;
    layer_3.row_i_number_of_entries = layer_2.output_row_i_number_of_entries;
    layer_3.execute(
            .clk                    (clk),
            .data_bus               (data_bus),
            .weight_bus             (weight_bus),
            .control_bus            (control_bus),
            .next_command_interrupt (next_command_interrupt),
            .output_line_stored     (output_line_stored)
        );
    layer_3.validate_outputs();

        
    $display("execute layer_4");
    layer_4.row_i = layer_3.output_row_i;
    // layer_4.row_i = layer_3.ground_truth_output_row_i;
    layer_4.row_i_number_of_entries = layer_3.output_row_i_number_of_entries;
    layer_4.execute(
            .clk                    (clk),
            .data_bus               (data_bus),
            .weight_bus             (weight_bus),
            .control_bus            (control_bus),
            .next_command_interrupt (next_command_interrupt),
            .output_line_stored     (output_line_stored)
        );
    layer_4.validate_outputs();

        
    $display("execute layer_5");
    layer_5.row_i = layer_4.output_row_i;
    // layer_5.row_i = layer_4.ground_truth_output_row_i;
    layer_5.row_i_number_of_entries = layer_4.output_row_i_number_of_entries;
    layer_5.execute(
            .clk                    (clk),
            .data_bus               (data_bus),
            .weight_bus             (weight_bus),
            .control_bus            (control_bus),
            .next_command_interrupt (next_command_interrupt),
            .output_line_stored     (output_line_stored)
        );
    layer_5.validate_outputs();

        
    $display("execute layer_6");
    layer_6.row_i = layer_5.output_row_i;
    // layer_6.row_i = layer_5.ground_truth_output_row_i;
    layer_6.row_i_number_of_entries = layer_5.output_row_i_number_of_entries;
    layer_6.execute(
            .clk                    (clk),
            .data_bus               (data_bus),
            .weight_bus             (weight_bus),
            .control_bus            (control_bus),
            .next_command_interrupt (next_command_interrupt),
            .output_line_stored     (output_line_stored)
        );
    layer_6.validate_outputs();

        
    $display("execute layer_7");
    layer_7.row_i = layer_6.output_row_i;
    // layer_7.row_i = layer_6.ground_truth_output_row_i;
    layer_7.row_i_number_of_entries = layer_6.output_row_i_number_of_entries;
    layer_7.execute(
            .clk                    (clk),
            .data_bus               (data_bus),
            .weight_bus             (weight_bus),
            .control_bus            (control_bus),
            .next_command_interrupt (next_command_interrupt),
            .output_line_stored     (output_line_stored)
        );
    layer_7.validate_outputs();

        
    $display("execute layer_8");
    layer_8.row_i = layer_7.output_row_i;
    // layer_8.row_i = layer_7.ground_truth_output_row_i;
    layer_8.row_i_number_of_entries = layer_7.output_row_i_number_of_entries;
    layer_8.execute(
            .clk                    (clk),
            .data_bus               (data_bus),
            .weight_bus             (weight_bus),
            .control_bus            (control_bus),
            .next_command_interrupt (next_command_interrupt),
            .output_line_stored     (output_line_stored)
        );
    layer_8.validate_outputs();

        
    $display("execute layer_9");
    layer_9.row_i = layer_8.output_row_i;
    // layer_9.row_i = layer_8.ground_truth_output_row_i;
    layer_9.row_i_number_of_entries = layer_8.output_row_i_number_of_entries;
    layer_9.execute(
            .clk                    (clk),
            .data_bus               (data_bus),
            .weight_bus             (weight_bus),
            .control_bus            (control_bus),
            .next_command_interrupt (next_command_interrupt),
            .output_line_stored     (output_line_stored)
        );
    layer_9.validate_outputs();

        
    $display("execute layer_10");
    layer_10.row_i = layer_9.output_row_i;
    // layer_10.row_i = layer_9.ground_truth_output_row_i;
    layer_10.row_i_number_of_entries = layer_9.output_row_i_number_of_entries;
    layer_10.execute(
            .clk                    (clk),
            .data_bus               (data_bus),
            .weight_bus             (weight_bus),
            .control_bus            (control_bus),
            .next_command_interrupt (next_command_interrupt),
            .output_line_stored     (output_line_stored)
        );
    layer_10.validate_outputs();

        
    $display("execute layer_11");
    layer_11.row_i = layer_10.output_row_i;
    // layer_11.row_i = layer_10.ground_truth_output_row_i;
    layer_11.row_i_number_of_entries = layer_10.output_row_i_number_of_entries;
    layer_11.execute(
            .clk                    (clk),
            .data_bus               (data_bus),
            .weight_bus             (weight_bus),
            .control_bus            (control_bus),
            .next_command_interrupt (next_command_interrupt),
            .output_line_stored     (output_line_stored)
        );
    layer_11.validate_outputs();

        
    $display("execute layer_12");
    layer_12.row_i = layer_11.output_row_i;
    // layer_12.row_i = layer_11.ground_truth_output_row_i;
    layer_12.row_i_number_of_entries = layer_11.output_row_i_number_of_entries;
    layer_12.execute(
            .clk                    (clk),
            .data_bus               (data_bus),
            .weight_bus             (weight_bus),
            .control_bus            (control_bus),
            .next_command_interrupt (next_command_interrupt),
            .output_line_stored     (output_line_stored)
        );
    layer_12.validate_outputs();

        
    $display("execute layer_13");
    layer_13.row_i = layer_12.output_row_i;
    // layer_13.row_i = layer_12.ground_truth_output_row_i;
    layer_13.row_i_number_of_entries = layer_12.output_row_i_number_of_entries;
    layer_13.execute(
            .clk                    (clk),
            .data_bus               (data_bus),
            .weight_bus             (weight_bus),
            .control_bus            (control_bus),
            .next_command_interrupt (next_command_interrupt),
            .output_line_stored     (output_line_stored)
        );
    layer_13.validate_outputs();

        