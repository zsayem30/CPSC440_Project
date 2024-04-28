function [mean_percentage_error, std_percentage_error] = prev_val(all_data, validation_output, errors)

%Compute the percentage error in the predictionss. This is equal to the
%euclidean error (distance) between prediction and actual divided by the euclidean distance
%between the actual data and the data in the previous timestamp.

        previous_values = zeros(size(validation_output, 1), 3);

        for n = 1:size(validation_output, 1)
            curr_index = find(all_data(:, 1) == validation_output(n, 1));
            if(curr_index == 1)
                prev_value = [0, 0, 0];
            else
                prev_index = curr_index - 1;
                prev_value = all_data(prev_index, 2:4);
            end
            previous_values(n, 1:3) = prev_value;
        end 

        displacement_Vector = validation_output(:, 2:4) - prev_values;
        distance_travelled = sqrt(displacement_Vector(:, 1).^2 + displacement_Vector(:, 2).^2 + displacement_Vector(:, 3).^2);
        non_zero_indices = find(~all(distance_travelled==0,2));

        percentage_error = (errors(non_zero_indices, :)./distance_travelled(non_zero_indices, :))*100;
        mean_percentage_error = round(mean(percentage_error), 9);
        std_percentage_error = round(std(percentage_error), 9);
