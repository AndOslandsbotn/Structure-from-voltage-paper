%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
view1 = 20
view2 = -25
fontsize = 30;
axis_number_size = 15;

color_red = '#CD3333'
color_yel = '#E3CF57'
color_gray = '#C1CDCD'
color_darkgray = '#838B8B'

ms_ll = 3; % Marker size on longitudes and latitudes
ms_surf = 3 % Marker size on surface

nlm = 7;
theta_max = 90;
phi_max = 180;
%exp_name='MnistNlmPerDigit1_expnr1'
exp_name = strcat('SphereTheta', num2str(theta_max), 'Phi', num2str(phi_max), 'nlm', num2str(nlm),'_ss');
%folder = strcat('ResultsPaper/', exp_name);
folder = strcat('Results/', exp_name);

save_figures = strcat('Figures2/', exp_name)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

filename = 'mds_emb';
mds_emb = table2array(readtable(fullfile(folder, filename)));

filename = 'v_emb';
v_emb = table2array(readtable(fullfile(folder, filename)));

filename = 'sphere_segment';
ss = table2array(readtable(fullfile(folder, filename)));

filename = 'idx_long';
idx_long = readmatrix(fullfile(folder, filename));
idx_long = idx_long+1; % Since python have different indexing than matlab

filename = 'idx_lat';
idx_lat = readmatrix(fullfile(folder, filename));
idx_lat = idx_lat+1; % Since python have different indexing than matlab

filename = 'idx_lm';
idx_lm = csvread(fullfile(folder, filename));
idx_lm(idx_lm==0) = nan
idx_lm = idx_lm +1;

[xs_long, ys_long, zs_long, xe_long, ye_long, ze_long] = points_on_long(ss, mds_emb, idx_long);
[xs_lat, ys_lat, zs_lat, xe_lat, ye_lat, ze_lat] = points_on_lat(ss, mds_emb, idx_lat);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MDS Embedding sphere scatter %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
maxs = max(mds_emb, [], 1);
mins = min(mds_emb, [], 1);
eps = 0.5;
[xq, yq, zq] = make_mesh(mds_emb, maxs, mins, eps);
fig4 = figure(4)
scatter(mds_emb(:, 2), mds_emb(:, 3), ms_surf, 'MarkerFaceColor', color_gray, 'MarkerEdgeColor', color_gray);
hold on
[numsources, m] = size(idx_lm);
for i=1:numsources
    idx = rmmissing(idx_lm(i, :));
    if i == 1
        scatter(mds_emb(idx, 2), mds_emb(idx, 3), 'MarkerFaceColor', color_red, 'MarkerEdgeColor', color_red);
    else
        scatter(mds_emb(idx, 2), mds_emb(idx, 3), 'MarkerFaceColor', color_yel, 'MarkerEdgeColor', color_yel);
    end
end

n = size(xe_long, 1);
for i=1:n
    h1 = plot(ye_long{i}, ze_long{i},  'LineWidth', 1, 'color', color_darkgray);
    h1 = scatter(ye_long{i}, ze_long{i}, ms_ll, 'MarkerFaceColor', color_darkgray, 'MarkerEdgeColor', color_darkgray);
end

n = size(xe_lat, 1);
for i=1:n
    h2 = plot(ye_lat{i}, ze_lat{i},  'LineWidth', 1, 'color', color_darkgray);
    h2 = scatter(ye_lat{i}, ze_lat{i},  ms_ll, 'MarkerFaceColor', color_darkgray, 'MarkerEdgeColor', color_darkgray);
end
uistack(h1, 'top')
%uistack(h2, 'top')
xlabel('x-axis', 'FontSize', fontsize);
ylabel('y-axis', 'FontSize', fontsize);
%xlim([mins(1), maxs(1)]);
%ylim([mins(3), maxs(3)]);
%xlim([-0.82, 0.52]);
%ylim([0.28,  0.92]);

%xticks([-0.8, 0, 0.5]);
%yticks([0.3, 0.0, 0.9]);

%if nlm == 3;
%xticks([-0.4, 0, 0.4]);
%yticks([0.2, 0, 1.0]);
%elseif nlm == 5;
%yticks([0.3, 0.5, 0.7]);
%elseif nlm == 7;
%yticks([-0.2, 0.0, 0.4]);
%end

%view(view1, view2);
grid off
%ax = gca;
%ax.ZGrid = 'off';
set(gca,'FontSize', axis_number_size);
savefig(strcat(save_figures, 'MDSemb2d'));
saveas(fig4, strcat(save_figures, 'MDSemb2d'), 'epsc');
saveas(fig4, strcat(save_figures, 'MDSemb2d'), 'pdf');

function [xs, ys, zs, xe, ye, ze] = points_on_long(surface, embedding, indices)
    [n, m] = size(indices);
    
    xs = cell(n, 1);
    ys = cell(n, 1);
    zs = cell(n, 1);
    
    xe = cell(n, 1);
    ye = cell(n, 1);
    ze = cell(n, 1);
    for i=1:n
        idx = rmmissing(indices(i, :));
        
        long_surf = surface(idx,:);
        long_embedding = embedding(idx, :);
    
        [zs{i}, sort_idx]=sort(long_surf(:, 3));
        xs{i}=long_surf(sort_idx, 1);
        ys{i}=long_surf(sort_idx, 2);
    
        xe{i} = long_embedding(sort_idx, 1);
        ye{i} = long_embedding(sort_idx, 2);
        ze{i} = long_embedding(sort_idx, 3);
    end
end

function [xs, ys, zs, xe, ye, ze] = points_on_lat(surface, embedding, indices)
    [n, m] = size(indices);
    
    xs = cell(n, 1);
    ys = cell(n, 1);
    zs = cell(n, 1);
    
    xe = cell(n, 1);
    ye = cell(n, 1);
    ze = cell(n, 1);
    for i=1:n
        idx = rmmissing(indices(i, :));
        
        lat_surf = surface(idx,:);
        lat_embedding = embedding(idx, :);
    
        [xs{i}, sort_idx]=sort(lat_surf(:, 1));
        ys{i}=lat_surf(sort_idx, 2);
        zs{i}=lat_surf(sort_idx, 3);
    
        xe{i} = lat_embedding(sort_idx, 1);
        ye{i} = lat_embedding(sort_idx, 2);
        ze{i} = lat_embedding(sort_idx, 3);
    end
end


function [xq, yq, vq] = make_mesh(x, maxs, mins, eps)
    [xq,yq] = meshgrid(mins(1)-eps:.001:maxs(1)+eps, mins(2)-eps:.001:maxs(2)+eps);
    vq = griddata(x(:, 1), x(:, 2), x(:, 3), xq, yq);
end