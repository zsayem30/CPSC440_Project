function [avg_pos_err, std_pos_err, euclidean_errs, avg_euclidean_err, std_euclidean_err, orient_errs, avg_orient_err, std_orient_err] = calculate_errors(actual_output, predicted_output)
%Computes the average error in each individual axis, average euclidean
%error and average orientation error between the predicted outputs and the
%actual outputs.
        errors = abs(predicted_output(:, 2:4) - actual_output(:, 2:4));

        avg_pos_err = round(nanmean(errors, 1), 9) * 1000;
        std_pos_err = round(nanstd(errors, 0, 1), 9) * 1000;

        euclidean_errs = sqrt(errors(:, 1).^2 + errors(:, 2).^2 + errors(:, 3).^2);
        avg_euclidean_err = round(mean(euclidean_errs), 9) * 1000;
        std_euclidean_err = round(std(euclidean_errs), 9) * 1000;

        if size(predicted_output, 2) > 4
            gmr_quat = quaternion(predicted_output(:, 5:8));
            ecm_quat = quaternion(actual_output(:, 5:8));
            angle_between = rad2deg(dist(gmr_quat, ecm_quat));
            avg_orient_err = round(nanmean(angle_between), 5);
            std_orient_err = round(nanstd(angle_between), 5);
        end
        euclidean_errs = mat2cell(euclidean_errs.', 1);
        orient_errs = mat2cell(angle_between.', 1);
end