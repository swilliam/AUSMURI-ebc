%% Plot EBC maps
% Plot figures tightly and using spikes
% Yanbo Lian 20210913

close all
clc
clear

file_name = 'prightwidevismright100vis90tI2.mat'; % file of the data
load(file_name) % load trained model
mr = max(resp);
%file_name = 'pboxmbox100vis90tC5.mat'; % file of the data
%load(file_name) % load trained model

%data_file_name = 'boxmazedat_125cm.mat';
%load(data_file_name, 'hds_data', 'positions_data') % load head directions and positions

readme = 'Positions (x,y) are in cm. The origin (0,0) is the bottom left point of the environment. Head directions are in radian. The East is 0 and the North (up) is pi/2.';

positions = pos(2:end,:) / 10; % Unit: cm
hds = md(2:end) + 90; % East is 0 degree
delta_t = 0.03; % units: second
Nt = length(hds);

peak_rate = 30; % Set the peak firing rate of the whole population
S_test =  peak_rate*resp(2:end,:)'./mr'; % scale the response

start_fig_number = 0;
n_rows = 2;% when plotting figures in the appendix

cells = 1 : 100; % All cells
cells = [23 54]; % proximal
% cells = [9 29]; % distal
% cells = [24 55]; % inverse
prox = [23 61];
distal = [37 57];
inverse = [ 24 55];
ebc_type = 'inverse'
i = 0;
for i_cell = inverse;
    i = i + 1;
    i_fig = start_fig_number + ceil(i/n_rows);
    i_row = mod(i, n_rows);
    
    if i_row == 1
        fig = figure(i_fig); 
        fig.Units = 'normalized'; fig.Position = [0 0 0.22 0.25]; % figures
%         fig.Units = 'normalized'; fig.Position = [0 0 0.15 0.85]; % figures in the appendix
        [ha, ~] = tight_subplot(n_rows, 3, [.01 .03],[.01 .01],[.01 .01]);
        i_plot = 1;
    end
    
    if i_row == 0
        i_row = n_rows;
    end
    
    ax = ha(i_plot); i_plot = i_plot + 1;
    axes(ax); 
    X = positions(:,1:2); % unit: cm
    x = linspace(0,125,round(125/3));
    occ = hist2w(X,ones(size(positions(:,1))),x,x)';
    vf = hist2w(X,S_test(i_cell,:),x,x)';
    rate_map = vf./occ;
    rate_map(isnan(rate_map)) =0;
    


%     rate_map = reshape(lca.rate_maps_smoothed(:,i_cell),env.Ny,env.Nx);
    rate_map = imresize(rate_map, 3, 'lanczos3');
%     rate_map = rate_map / max(rate_map(:));
    imagesc(rate_map);
    colormap(ax, 'parula'); 
    axis image
    axis off
    set(ax, 'YDir', 'normal')
    
    [thresh,L,U,C]= isoutlier(S_test(i_cell,:),'percentiles',[0 95]);
    %S_test(i_cell,~thresh)=0;
    %S_test(i_cell,:)= S_test(i_cell,:)/max(S_test(i_cell,:));

    % make sure it's column vector
    root(i_cell).x = positions(:,1); % unit: cm
    root(i_cell).y = positions(:,2); % unit: cm
    root(i_cell).ts = delta_t * (1:Nt)';
    root(i_cell).md = deg2rad(mod(hds,360))'; % movement (or head) direction: in radians
    %root(i_cell).spike = ((S_test(i_cell,:)) > U)'; % Transform rates to spikes
    root(i_cell).spike = ((S_test(i_cell,:)) > peak_rate*rand(1,Nt))'; % Transform rates to spikes

    % Plot EBC
    out = EgocentricRatemap(root(i_cell)); % where r is your behavioral/ephys struct
    i_plot = plotEBC_tight(root(i_cell), out, ha, i_plot);
    if i_row == 10
        saveas(gcf,[ 'simulwidecells ' num2str(i-9) '-' num2str(i) '.png'])
    elseif i_row == 2;
        saveas(gcf,[ 'cells_' ebc_type '.png'])
    end
end

%save('Simon_simulspike_data', 'root', 'readme');