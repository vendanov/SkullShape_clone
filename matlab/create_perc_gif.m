function create_perc_gif(input_mat_file, face_perc_file, input_obj_file, output_gif_file)
    % Load the energies
    load(input_mat_file);
    %energies = transpose(energies);

    % Load the mesh
    mesh = readObj(input_obj_file);


    % load face verticces
    load(face_perc_file);

    % create colormap basded on energies
    colormap_values = ones(size(energies));
    colormap_values(energies > perc75) = 2; % Green
    colormap_values(energies > perc95) = 3; % Red
    colormap_values(energies > perc99) = 4; % Red


    fig = figure;
    trisurf(mesh.f.v, mesh.v(:, 1), mesh.v(:, 2), mesh.v(:, 3), 'facevertexcdata', transpose(colormap_values) );

    colormap([0 0 1;0 1 0; 1 1 0; 1 0 0; ]);  %  Blue Green, Yellow, Red
    colorbar('Ticks', [1, 2, 3, 4], 'TickLabels', {'< 75%', '> 75%', '> 95%', '> 99%'});  % Display the colorbar with labels
    %colorbar;         % Display the colorbar
    axis equal;





    % Rotate and save frames
    num_frames = 10; % Number of frames in the GIF
    rotation_angle = 360 / num_frames;

    % Create a cell array to store frames
    frames = cell(1, num_frames);

    for frame_idx = 1:num_frames
        % Rotate the view
        fprintf('print new subject \n')
        view(30 + frame_idx * rotation_angle, 30);

        % Capture the current plot as a frame
        frames{frame_idx} = getframe(fig);

        % Convert the captured frame to an RGB image
        [imind, cm] = rgb2ind(frames{frame_idx}.cdata, 256);
        fprintf('print starting \n')
        % Write the image to the GIF file
        if frame_idx == 1
            imwrite(imind, cm, output_gif_file, 'gif', 'LoopCount', Inf, 'DelayTime', 1);
        else
            imwrite(imind, cm, output_gif_file, 'gif', 'WriteMode', 'append', 'DelayTime', 1);
        end
        fprintf('finish slide \n')
    end


    % Close the figure
    close(fig);

    disp('GIF saved successfully.');
end



