function create_perc_fig(input_mat_file,face_perc_file, input_obj_file, output_fig_file)
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
    savefig(fig,output_fig_file);

end



