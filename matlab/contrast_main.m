%% Plot EBC maps
% Plot figures tightly and using spikes
% Yanbo Lian 20210913

close all
clc
clear
contrasts = [0.5 0.2 0.1 0.05 0.02 0.01 0.005]
for c = contrasts
file_name = ['pexpt127widevismright100vis90tC' num2str(c) '.mat']; % file of the data
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
%cells = [23 54]; % proximal
% cells = [9 29]; % distal
% cells = [24 55]; % inverse
prox = [23 61];
distal = [37 57];
inverse = [ 24 55];
ebc_type = 'inverse'
i = 0;
for i_cell = cells;
    % make sure it's column vector
    root(i_cell).x = positions(:,1); % unit: cm
    root(i_cell).y = positions(:,2); % unit: cm
    root(i_cell).ts = delta_t * (1:Nt)';
    root(i_cell).md = deg2rad(mod(hds,360))'; % movement (or head) direction: in radians
    %root(i_cell).spike = ((S_test(i_cell,:)) > U)'; % Transform rates to spikes
    root(i_cell).spike = ((S_test(i_cell,:)) > peak_rate*rand(1,Nt))'; % Transform rates to spikes

 end

save(['Simon_spike_data_wide_expt127_' num2str(c) '.mat'], 'root', 'readme');
end