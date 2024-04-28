function cost = immitation_cost(yQuery, dataVal_outputs, ySigma) 
        %% Compute similarity and cost of imitation for points
        %Compute difference between GMR output and ECM actual output
        imm_diff = dataVal_outputs - yQuery;
        imm_diff(:, 1:3) = imm_diff(:, 1:3)*1000;
        similarity_Value = [];

        for r = 1:length(yQuery)
            [row, column] = find(isnan(imm_diff(r, :)));
            if isempty(unique(row))
                weight_matrix = squeeze(ySigma(r, :,:));
                similarity_Value(end+1) = imm_diff(r,:)*(weight_matrix\imm_diff(r, :).');
            end
        end

        similarity_Value = rmmissing(similarity_Value);
        cost = min(similarity_Value)*10^9;