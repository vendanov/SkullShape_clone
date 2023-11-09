function create_mah_fig(input_mat_file, input_obj_file, output_fig_file)
    % Load the energies
    load(input_mat_file);
    energies = transpose(energies);
    
    %energies(energies > 100) = 100;
    % Load the mesh
    mesh = readObj(input_obj_file);
    % ... rest of your code ...
    fig = figure;
    trisurf(mesh.f.v, mesh.v(:, 1), mesh.v(:, 2), mesh.v(:, 3), 'facevertexcdata', energies );
    colormap('jet');  % Set the colormap (adjust as desired)
    colorbar;         % Display the colorbar
    axis equal;
    clim([0,100]);
    savefig(fig,output_fig_file);

    % Close the figure
    close(fig);

    disp('GIF saved successfully.');
end



