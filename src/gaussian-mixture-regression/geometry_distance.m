function [geom_dist, geom_dist_avg, geom_dist_std, Area_Percent, Width_Percent, Height_Percent] = geometry_distance(validation_output, predicted_output, depth_distance, magnification_Factor) 
%% Compute Geometry distance
        geom_dist = [];       
        for j = 1:size(predicted_output, 1)

           syms x y z t

           actual_quat = validation_output(j, 5:8);
           actual_pos = validation_output(j, 2:4);
           actual_rotm = quat2rotm(actual_quat);
           actual_tform = trvec2tform(actual_pos);
           actual_tform(1:3, 1:3) = actual_rotm;
           
           %Find equation of the plane for the actual camera at a depth of
           %0.25 m.
%            depth_distance = 0.225;
%            magnification_Factor = 12.5;

           depth = depth_distance/magnification_Factor;
           
           actual_depth_plane_cam = [0, 0, depth, 1];
           actual_depth_plane_wrld = actual_tform * actual_depth_plane_cam.';
           actual_plane_norm = actual_depth_plane_wrld(1:3) - actual_pos;

           Plane_Eqn = actual_plane_norm(1)*(x - actual_depth_plane_wrld(1)) + actual_plane_norm(2) * (y - actual_depth_plane_wrld(2)) + actual_plane_norm(3)* (z - actual_depth_plane_wrld(3)) == 0;
           
           pred_quat = predicted_output(j, 5:8);
           pred_pos = predicted_output(j, 2:4);
           pred_rotm = quat2rotm(pred_quat);
           pred_tform = trvec2tform(pred_pos);
           pred_tform(1:3, 1:3) = pred_rotm;

           pred_depth_plane_cam = [0 0 depth 1];
           pred_depth_plane_wrld = pred_tform * pred_depth_plane_cam.';
           pred_plane_vec = pred_depth_plane_wrld(1:3) - pred_pos;

           xl = pred_pos(1) + pred_plane_vec(1)*t;
           yl = pred_pos(2) + pred_plane_vec(2)*t;
           zl = pred_pos(3) + pred_plane_vec(3)*t;

           Plane_Eqn = subs(Plane_Eqn, {x y z}, {xl yl zl});
           t = solve(Plane_Eqn, t);

           x_intersect = pred_pos(1) + pred_plane_vec(1)*t;
           y_intersect = pred_pos(2) + pred_plane_vec(2)*t;
           z_intersect = pred_pos(3) + pred_plane_vec(3)*t;

           intersect_point = [double(x_intersect), double(y_intersect), double(z_intersect)];

           %Compute distance from center of camera plane
           diff_intersect = intersect_point - actual_depth_plane_wrld(1:3).';
           dist_intersect = sqrt(diff_intersect(:, 1).^2 + diff_intersect(:, 2).^2 + diff_intersect(:, 3).^2);
           geom_dist(end+1) = dist_intersect;
        end
        
        %% Compute mean geometric distance in mm

        geom_dist_avg = round(mean(geom_dist), 9);
        geom_dist_std = round(std(geom_dist), 9);
        geom_dist = mat2cell(geom_dist, 1);

        %% Do percentage computations
        aspect_ratio_Width = 5;
        aspect_Ratio_Height = 4;
        aspect_Ratio_Hyp = sqrt(aspect_ratio_Width^2 + aspect_Ratio_Height^2);
        Monitor_Size = 0.4318; % 17 inches to meters
        Monitor_Width = (Monitor_Size / aspect_Ratio_Hyp) * aspect_Ratio_Height;
        Monitor_Height = (Monitor_Size / aspect_Ratio_Hyp) * aspect_ratio_Width;
        Monitor_Area = Monitor_Width * Monitor_Height;
        Area_Percent = ((pi * (geom_dist_avg)^2)/Monitor_Area)*100;
        Width_Percent = ((geom_dist_avg)/ Monitor_Width)*100;
        Height_Percent = ((geom_dist_avg)/ Monitor_Height)*100;

end