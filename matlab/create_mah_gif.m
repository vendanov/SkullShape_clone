function create_mah_gif(input_mat_file, input_obj_file, output_gif_file)
    % Load the energies
    load(input_mat_file);
    energies = transpose(energies);
    energies(energies > 100) = 100;
    % Load the mesh
    mesh = readObj(input_obj_file);
    % ... rest of your code ...
    fig = figure;
    trisurf(mesh.f.v, mesh.v(:, 1), mesh.v(:, 2), mesh.v(:, 3), 'facevertexcdata', energies );
    colormap('jet');  % Set the colormap (adjust as desired)
    colorbar;         % Display the colorbar
    axis equal;


    % Rotate and save frames
    num_frames = 10; % Number of frames in the GIF
    rotation_angle = 360 / num_frames;

    % Create a cell array to store frames
    frames = cell(1, num_frames);

    for frame_idx = 1:num_frames
        % Rotate the view
        view(30 + frame_idx * rotation_angle, 30);

        % Capture the current plot as a frame
        frames{frame_idx} = getframe(fig);

        % Convert the captured frame to an RGB image
        [imind, cm] = rgb2ind(frames{frame_idx}.cdata, 256);

        % Write the image to the GIF file
        if frame_idx == 1
            imwrite(imind, cm, output_gif_file, 'gif', 'LoopCount', Inf, 'DelayTime', 1);
        else
            imwrite(imind, cm, output_gif_file, 'gif', 'WriteMode', 'append', 'DelayTime', 1);
        end
    end


    % Close the figure
    close(fig);

    disp('GIF saved successfully.');
end



