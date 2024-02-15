% Before running this script, the qpOASES Matlab interface must be
% compiled. To do this, run ./Resources/qpOASES-3.1.0/interfaces/matlab/make.m

% SP=1;
% sys = TF;
% [y,t]=step(SP*sys); %get the response of the system to a step with amplitude SP
% sserror=abs(SP-y(end)) 

% clear all
% close all

addpath('./Resources')
addpath('./Resources/qpOASES-3.1.0/interfaces/matlab') 

%% Model
%rng(115123)
n = 4; % Number of states
m = 1; % Number of control inputs
% f_u = @dyn_motor_scaled; % Changed to immunotherapy cancer dynamics
f_uu = @dyn_motor_scaled_nominal; % nominal model
f_u = @dyn_motor_scaled; % Changed to immunotherapy cancer dynamics
time_interval = 5; % Interval 4 days
Tpred = 5;
Tmax = 100;

% Discretize, sampling time 0.5s
%% Par setting
u_scale = 6.4;
deltaT = 0.01;

interval = time_interval/deltaT -1 ;  % 4 days for simulation and collecting data
% x_ini = [0; 0; 0; 1.2];

x_ini = [0.1; 0.1; 0; 0.8];


% deltaT = 0.05;
% Prediction horizon
n_clusters = 10; % number of basis using kmeans
Np = round(Tpred / deltaT);

% Ntraj = 2; 
Ntraj = 100; 
% Nsim = 400; % deltaT*Nsim = 40 days per trajectory
% Tmax = 50; % Simlation legth
Nsim = Tmax/deltaT;

% Nrbf = 5;  % basis function

% Q = 1e1; % Weight matrices
% R = 0.1;
R = 0.1;
Q = 1e3; % P = Q = 1000
%% Model discretize
% Runge-Kutta 4
% nominal model
k11 = @(t,x,u) ( f_uu(t,x,u) );
k22 = @(t,x,u) ( f_uu(t,x + k11(t,x,u)*deltaT/2,u) );
k33 = @(t,x,u) ( f_uu(t,x + k22(t,x,u)*deltaT/2,u) );
k44 = @(t,x,u) ( f_uu(t,x + k11(t,x,u)*deltaT,u) );
f_udd = @(t,x,u) ( x + (deltaT/6) * ( k11(t,x,u) + 2*k22(t,x,u) + 2*k33(t,x,u) + k44(t,x,u)  )  + [1,0,0,0]' * u  );

% (Uncertain model)
k1 = @(t,x,u, seed) ( f_u(t,x,u, seed) );
k2 = @(t,x,u, seed) ( f_u(t,x + k1(t,x,u, seed)*deltaT/2,u, seed) );
k3 = @(t,x,u, seed) ( f_u(t,x + k2(t,x,u, seed)*deltaT/2,u, seed) );
k4 = @(t,x,u, seed) ( f_u(t,x + k1(t,x,u, seed)*deltaT,u, seed) );
f_ud = @(t,x,u, seed) ( x + (deltaT/6) * ( k1(t,x,u, seed) + 2*k2(t,x,u, seed) + 2*k3(t,x,u, seed) + k4(t,x,u, seed)  )  + [1,0,0,0]' * u  );


%% Collect data
rng(115123)
disp('Starting data collection')

Cy = [0 0 0 1]; % Output matrix: y = Cy*x
nD = 2; % Number of delays
ny = size(Cy,1); % Number of outputs

%% Random control input forcing and Random initial condition
Ubig = u_scale*rand(Nsim, Ntraj) - 0; % [0,7]
U_input = zeros(size(Ubig));
U_input(1:interval:end,:) = Ubig(1:interval:end,:);
Ubig = U_input;
% Ubig = [Ubig(2:end), 0];

Xcurrent = (rand(n,Ntraj)*1 - 0); % [0,1]  state x trajec/ Random initial states
% Xcurrent = repelem(x_ini,1,Ntraj);  % Initial conditions 

X = []; Y = []; U = [];
zeta_current = [ Cy*Xcurrent ; NaN(nD*(m + ny),Ntraj) ];  % [1; 2 * (in + out)]  (dim of embedded states)

% Delay-embedded "state" , lifted state and control input
% zeta_k = [y_{k} ; u_{k-1} ; y_{k-1} ... u_{k-nd} ; y_{k-nd} ];  
n_zeta = (nD+1)*ny + nD*m; % dimension of the delay-embedded "state": [(D+1)*out + D*in]
% num of states + num of inputs
%% Collect data
% Nsim = 4;
for i = 1:Nsim
%     Xnext = f_ud( 0, Xcurrent, Ubig(i,:) ) + Ubig(i,:);  % Random control input for simulation
    Xnext = f_udd( 0, Xcurrent, Ubig(i,:) ); % Random control input for simulation
    
    zeta_prev = zeta_current;
    zeta_current = [[ Cy*Xnext ; Ubig(i,:)] ; zeta_current( 1:end-ny-m , : ) ];
    if(i > nD)
        X = [X zeta_prev];
        Y = [Y zeta_current];
        U = [U Ubig(i,:)];
    end
    Xcurrent = Xnext;
end
fprintf('Data collection DONE \n');


%% Basis
%% RBF centers:
% 1. preprocess the time series data
X_time_seris = Y';   % n_zeta * num_traject
X_norm = (X_time_seris - mean(X_time_seris)) ./ std(X_time_seris);

% 2. define the number of clusters
% n_clusters = round(sqrt(size(X_time_seris,1) * size(X_time_seris,2)));

% 3. select the clustering algorithm and fit it to the data
[idx, centers] = kmeans(X_norm, n_clusters);

% 4. obtain the cluster labels
labels = idx;

% 5. calculate the cluster centers
centers = zeros(n_clusters, size(X_time_seris, 2));
for i = 1:n_clusters
    centers(i, :) = mean(X_time_seris(labels == i, :), 1);
end

% 6. use the cluster centers as the centers of Gaussian RBF
rbf_centers = centers';

rbf_type = 'thinplate';
% rbf_type = 'gauss';
% theta_max = pi;
liftFun = @(xx)( [xx;rbf(xx,rbf_centers,rbf_type)] );

Nrbf = n_clusters; % number of radial basis function
Nlift = Nrbf + n_zeta;


%% Lift
disp('Starting LIFTING')

% rbf(X,cent,rbf_type)

Xlift = liftFun(X);
Ylift = liftFun(Y);

%% Numerical A,B,C matrix by Regression
disp('Starting REGRESSION for A,B,C')
W = [Ylift ; X];
V = [Xlift ; U];

WVt = W*V';
VVt = V*V';
ABC = WVt * pinv(VVt);

Alift = ABC(1:Nlift,1:Nlift);
Blift = ABC(1:Nlift,Nlift+1:end);
Clift = ABC(Nlift+1:end,1:Nlift);
fprintf('Regression for A, B, C DONE \n');

% Residual
fprintf( 'Regression residual %f \n', norm(Ylift - Alift*Xlift - Blift*U,'fro') / norm(Ylift,'fro') );


%% ************************* Predictor comparison *************************
% uprbs = (1*myprbs(Nsim,0.5) - 0);
uprbs = ( u_scale*rand(Nsim,1) - 0);
uprb_in = zeros(size(uprbs));
uprb_in(1:interval:end,:) = uprbs(1:interval:end,:);
uprbs = uprb_in;
% uprbs = [uprbs(2:end)',0]';
u_dt = @(i)(  uprbs(i+1) );

x0 = x_ini;  % Initial conditions
x = x0;

% Delayed initial condition (assume random control input in the past)
xstart = [Cy*x ; NaN(nD*(ny+m),1)];
for i = 1:nD
    urand = u_scale*rand(m,1) - 0;
    xp = f_udd(0,x,urand);
    xstart = [Cy*xp ; urand; xstart(1:end-ny-m)];
%     x = xp;
end

%% Inital conditions
x_true = xp;
xlift = liftFun(xstart);

X = xp;
U=[]; 

XXX = xp;

% Local linearization
xloc = xp;
x = sym('x',[4 1]); syms u;
Aloc = double(subs(jacobian(f_udd(0,x,u),x),[x;u],[xloc;urand]));
Bloc = double(subs(jacobian(f_udd(0,x,u),u),[x;u],[xloc;urand]));
cloc = double(subs(f_udd(0,x,u),[x;u],[xloc;urand])) - Aloc*xloc - Bloc*urand;



%% Simulation
for i = 0:Nsim-1
%     i=2;
    % True dynamics
    x_true = [x_true, f_udd(0,x_true(:,end),u_dt(i)   ) ];
    
    % Koopman predictor
    xlift = [xlift Alift*xlift(:,end) + Blift*u_dt(i)];
    
    % Local linearization predictor
    xloc = [xloc Aloc*xloc(:,end) + Bloc*u_dt(i) + cloc];
end

figure
lw_koop = 3;
font_size = 23;
xlift_comparison = xlift;

plot([0:Nsim]*deltaT,Cy*x_true,'-b','linewidth', lw_koop); hold on
plot([0:Nsim]*deltaT,Clift(1,:)*xlift_comparison, '-r','linewidth',lw_koop); hold on
% plot([1:length(Cy * XXX)]*deltaT, Cy * z, '-r','linewidth',lw_koop); hold on

plot([0:Nsim]*deltaT,Cy*xloc, 'linewidth',lw_koop)

grid on
LEG = legend('True','Koopman','Local at $x_0$');
set(gca,'GridLineStyle','--')     % use ':' for dots
set(LEG,'Interpreter','latex','location','northeast','fontsize',font_size)

axis([0 Tmax -0.3 1])
% title("Predictor comparison")
xlabel("Time (days)")
ylabel("Tu (10^6 cells)")
set(gca,'FontSize',font_size);


%% Control part
%% ********************** Feedback control ********************************

Nsim = Tmax/deltaT;

REF = 'exp'; % 'step' or 'cos'
switch REF
    case 'exp'
        ymin = 0;
        ymax = 2;
%         x0 = [0.1;0.1;0;0.8];
%         yrr = x_ini(4) * exp(-1.5*([1:Nsim]) / Nsim); % reference
        yrr = zeros(size(x_ini(4) * exp(-1.5*([1:Nsim]) / Nsim))); % reference
%         yrr = [1.1 * ones(Nsim/2,1)', 0.2 * ones(Nsim/2,1)']; % reference
end



%% Build Koopman MPC controller
C = zeros(1,Nlift); C(1) = 1;

% Constraints
xlift_min = [nan(1,1) ; nan(Nlift-1,1)  ];
xlift_max = [nan(1,1) ; nan(Nlift-1,1)  ];
Ulb = 2.2; 
Uub = u_scale;
% Uub = 3;


QP = Q; % terminal cost
koopmanMPC  = getMPC(Alift, Blift, C, 0, Q, R, QP, Np, Ulb, Uub, xlift_min, xlift_max, 'qpoases');


% getMPC_delta_u(Alift, Blift, C, 0, Q, R, QP, Np, Ulb, Uub, xlift_min, xlift_max, 'qpoases');

% R = 0.1 ; 
% Q_delta_u = 10;

% koopmanMPC  = getMPC_delta_u(Alift, Blift, C, 0, Q, R, QP, Np, Ulb, Uub, xlift_min, xlift_max, 'qpoases',  'Q_delta_u', 100);


%% Certain Model Parameter (Nominal input)
%
%%
num_sim_ini_condition = 1;
for j = 1:num_sim_ini_condition
    %% Initial condition for the delay-embedded state (assuming zero control in the past)
    % rng(j+1111)

    x_ini = [0.1, 0.1, 0, 0.8]';
    
    x = x_ini;
    desr_end = 1e-2;
    % yrr = 0.8 * exp(log(desr_end/x_ini(4))/desr_time * ([1:Nsim]) * Tmax / Nsim  ); % reference

    %% 
    zeta0 = [Cy*x ; NaN(nD*(ny+m),1)];
    for i = 1:nD
        upast = zeros(m,1);

        % seed_initial = j;
        % xp = f_udd(0,x,upast,seed_initial);
        xp = f_udd(0,x,upast);

        zeta0 = [Cy*xp ; upast ; zeta0(1:end-ny-m)]; 
        x = xp;
    end
    x0 = x;
    
    x_koop = x0; x_loc = x0;
    zeta = zeta0; % Delay-embedded "state"
    
    XX_koop = x0; UU_koop = [];
    XX_loc = x0; UU_loc = [];
    
    
    
    %% Get Jacobian of the true dynamics (for local linearization MPC)
    x = sym('x',[4 1]); syms u;
    % f_ud_sym = f_ud(0,x,u,seed_initial);
    f_ud_sym = f_udd(0,x,u);
    u_loc = 0;
    Jx = jacobian(f_ud_sym,x);
    Ju = jacobian(f_ud_sym,u);
    
    wasinfeas= 0;
    ind_inf = [];
    
    
    %% Closed-loop simultion start:  MPC linearized vs koopman
    UU_koop = [];
    
    
    step = 1;
    flag = 0; % flag when terminal constraint is met for the first time

    for i = 0:step:Nsim-1
        % if(mod(i,10) == 1)
        %     fprintf('Closed-loop simulation: iterate %i out of %i \n', i, Nsim)
        % end
        
        % reference signal
        yr = yrr(i+1);
        
    
    %     up_num = Tpred/deltaT -1 ;  % 7 days
        up_num = time_interval/deltaT -1  ;  
        apply_interval = 0;
        
        
        % Koopman MPC
        xlift = liftFun(zeta); % Lift
    
        
        %% Control every "up_num" samples
        if( (mod(i,up_num) == apply_interval) && (Cy*x_koop > 1e-3))  %%  t = 10 compute input, apply it to t = 11
            u_koop = koopmanMPC(xlift,yr); % Get control input
    %     elseif(mod(i,10) == 1)
            % flag = 1;
        else
    %         u_koop = koopmanMPC(xlift,yr); % Get control input
            if((Cy*x_koop < 1e-3))
                flag = 1;
            end
    
            u_koop = 0;
        end
        
        if(flag==1)
            u_koop = 0;
        end
        
        %% State update

        seed = (j);
        x_koop = f_ud(0,x_koop,u_koop, seed); % Update true state
        zeta = [ Cy*x_koop ; u_koop; zeta(1:end-ny-m)]; % Update delay-embedded state
            
        XX_koop = [XX_koop x_koop];
        UU_koop = [UU_koop u_koop];
        
    end
    % B_record(j,:) = XX_koop(1,:);
    % E_record(j,:) = XX_koop(2,:);
    % Ti_record(j,:) = XX_koop(3,:);
    % Tu_record(j,:) = XX_koop(4,:);

    % T_u_end_nominal(j) = find(XX_koop(4,4000:end) <desr_end, 1 ) * deltaT  ;

    % if(isempty(ind_inf))
    %     ind_inf = Nsim;
    % end
    
end

UU_nominal = UU_koop;



% %%%% Uncertain Model Parameter (Uncertain measurement)
% %%
% 
% koopmanMPC  = getMPC(Alift, Blift, C, 0, Q, R, QP, Np, Ulb, Uub, xlift_min, xlift_max, 'qpoases');
% 
% x_ini_unchange = [0.1, 0.1, 0, 0.8]';
% % num_sim_ini_condition = 20;
% 
% num_sim_ini_condition = 200;
% 
% 
% % figure
% 
% len_sim = Nsim +1 ;
% 
% B_record = zeros(num_sim_ini_condition, len_sim );
% 
% E_record = zeros(num_sim_ini_condition, len_sim );
% 
% Ti_record = zeros(num_sim_ini_condition, len_sim );
% 
% Tu_record = zeros(num_sim_ini_condition, len_sim );
% 
% T_u_end_nominal = zeros(1,num_sim_ini_condition);
% 
% for j = 1:num_sim_ini_condition
%     %% Initial condition for the delay-embedded state (assuming zero control in the past)
%     rng(j+1111)
%     % x0 = 1*rand(4,1) ;
%     if j < num_sim_ini_condition
%         % x_ini = [1*rand(1,1), 1*rand(1,1), 1*rand(1,1), 1*rand(1,1)+1]' ;
%         % x_ini = x_ini + normrnd(0, x_ini * 0.2); %% noisy initial values
% 
%         x_ini = [0.1, 0.1, 0, 0.8]';
%         % x_ini = x_ini_unchange + unifrnd(-x_ini * 0.1, x_ini * 0.1); %% noisy initial values
%         % x_ini = x_ini_unchange + unifrnd(-x_ini / 1.1, x_ini / 0.9); %% noisy initial values
%         % x_ini = unifrnd(x_ini / 1.1, x_ini / 0.9); %% noisy initial values  
%     else
%         x_ini = [0.1, 0.1, 0, 0.8]';
%     end 
% 
%     % load('uncertain_initial.mat')
%     % x_ini = x_1(j,:)';
% 
%     % x = x_ini;
%     % x_ini = x_ini + normrnd(0, x_ini * 0.05); %% noisy initial values
%     x = x_ini;
%     desr_end = 1e-2;
%     % yrr = 0.8 * exp(log(desr_end/x_ini(4))/desr_time * ([1:Nsim]) * Tmax / Nsim  ); % reference
% 
%     %% 
%     zeta0 = [Cy*x ; NaN(nD*(ny+m),1)];
%     for i = 1:nD
%         upast = zeros(m,1);
% 
%         % seed_initial = j;
%         % xp = f_udd(0,x,upast,seed_initial);
%         xp = f_udd(0,x,upast);
% 
%         zeta0 = [Cy*xp ; upast ; zeta0(1:end-ny-m)]; 
%         x = xp;
%     end
%     x0 = x;
% 
%     x_koop = x0; x_loc = x0;
%     zeta = zeta0; % Delay-embedded "state"
% 
%     XX_koop = x0; UU_koop = [];
%     XX_loc = x0; UU_loc = [];
% 
% 
% 
%     %% Get Jacobian of the true dynamics (for local linearization MPC)
%     x = sym('x',[4 1]); syms u;
%     % f_ud_sym = f_ud(0,x,u,seed_initial);
%     f_ud_sym = f_udd(0,x,u);
%     u_loc = 0;
%     Jx = jacobian(f_ud_sym,x);
%     Ju = jacobian(f_ud_sym,u);
% 
%     wasinfeas= 0;
%     ind_inf = [];
% 
% 
%     %% Closed-loop simultion start:  MPC linearized vs koopman
%     UU_koop = [];
% 
% 
%     step = 1;
%     flag = 0; % flag when terminal constraint is met for the first time
% 
%     for i = 0:step:Nsim-1
%         % if(mod(i,10) == 1)
%         %     fprintf('Closed-loop simulation: iterate %i out of %i \n', i, Nsim)
%         % end
% 
%         % reference signal
%         yr = yrr(i+1);
% 
% 
%     %     up_num = Tpred/deltaT -1 ;  % 7 days
%         up_num = time_interval/deltaT -1  ;  
%         apply_interval = 0;
% 
% 
%         % Koopman MPC
%         xlift = liftFun(zeta); % Lift
% 
% 
%         %% Control every "up_num" samples
%         if( (mod(i,up_num) == apply_interval) && (Cy*x_koop > 1e-3))  %%  t = 10 compute input, apply it to t = 11
%             u_koop = koopmanMPC(xlift,yr); % Get control input
%     %     elseif(mod(i,10) == 1)
%             % flag = 1;
%         else
%     %         u_koop = koopmanMPC(xlift,yr); % Get control input
%             if((Cy*x_koop < 1e-3))
%                 flag = 1;
%             end
% 
%             u_koop = 0;
%         end
% 
%         if(flag==1)
%             u_koop = 0;
%         end
% 
%         %% State update
% 
%         seed = (j);
%         x_koop = f_ud(0,x_koop,u_koop, seed); % Update true state
%         zeta = [ Cy*x_koop ; u_koop; zeta(1:end-ny-m)]; % Update delay-embedded state
% 
%         XX_koop = [XX_koop x_koop];
%         UU_koop = [UU_koop u_koop];
% 
%     end
%     B_record(j,:) = XX_koop(1,:);
%     E_record(j,:) = XX_koop(2,:);
%     Ti_record(j,:) = XX_koop(3,:);
%     Tu_record(j,:) = XX_koop(4,:);
% 
%     T_u_end_nominal(j) = find(XX_koop(4,4000:end) <desr_end, 1 ) * deltaT  ;
% 
%     if(isempty(ind_inf))
%         ind_inf = Nsim;
%     end
% 
% end
% 
% 
% B_record_max = prctile(B_record, 97.5);
% B_record_min = prctile(B_record, 2.5);
% 
% E_record_max = prctile(E_record, 97.5);
% E_record_min = prctile(E_record, 2.5);
% 
% Ti_record_max = prctile(Ti_record, 97.5);
% Ti_record_min = prctile(Ti_record, 2.5);
% 
% Tu_record_max = prctile(Tu_record, 97.5);
% Tu_record_min = prctile(Tu_record, 2.5);
% 
% 
% if(isempty(ind_inf))
%     ind_inf = Nsim;
% end
% 
% % u_nominal = UU_koop;

%% Uncertain Model Parameter (nominal input)
%%

% koopmanMPC  = getMPC(Alift, Blift, C, 0, Q, R, QP, Np, Ulb, Uub, xlift_min, xlift_max, 'qpoases');

x_ini_unchange = [0.1, 0.1, 0, 0.8]';
% num_sim_ini_condition = 20;

num_sim_ini_condition = 200;


% figure

len_sim = Nsim +1 ;

B_record = zeros(num_sim_ini_condition, len_sim );

E_record = zeros(num_sim_ini_condition, len_sim );

Ti_record = zeros(num_sim_ini_condition, len_sim );

Tu_record = zeros(num_sim_ini_condition, len_sim );

T_u_end_nominal = zeros(1,num_sim_ini_condition);

for j = 1:num_sim_ini_condition
    %% Initial condition for the delay-embedded state (assuming zero control in the past)
    rng(j+1111)
    % x0 = 1*rand(4,1) ;
    if j < num_sim_ini_condition
        x_ini = [0.1, 0.1, 0, 0.8]';

        % x_ini = unifrnd(x_ini / 1.1, x_ini / 0.9); %% noisy initial values  
    else
        x_ini = [0.1, 0.1, 0, 0.8]';
    end 
    
    x = x_ini;
    desr_end = 1e-2;
    
    % yrr = 0.8 * exp(log(desr_end/x_ini(4))/desr_time * ([1:Nsim]) * Tmax / Nsim  ); % reference

    %% 
    zeta0 = [Cy*x ; NaN(nD*(ny+m),1)];
    for i = 1:nD
        upast = zeros(m,1);

        % seed_initial = j;
        % xp = f_udd(0,x,upast,seed_initial);
        xp = f_udd(0,x,upast);

        zeta0 = [Cy*xp ; upast ; zeta0(1:end-ny-m)]; 
        x = xp;
    end
    x0 = x;
    
    x_koop = x0; x_loc = x0;
    zeta = zeta0; % Delay-embedded "state"
    
    XX_koop = x0; UU_koop = [];
    XX_loc = x0; UU_loc = [];
    
    
    
    %% Get Jacobian of the true dynamics (for local linearization MPC)
    x = sym('x',[4 1]); syms u;
    % f_ud_sym = f_ud(0,x,u,seed_initial);
    f_ud_sym = f_udd(0,x,u);
    u_loc = 0;
    Jx = jacobian(f_ud_sym,x);
    Ju = jacobian(f_ud_sym,u);
    
    wasinfeas= 0;
    ind_inf = [];
    
    
    %% Closed-loop simultion start:  MPC linearized vs koopman
    UU_koop = [];
    
    
    step = 1;
    flag = 0; % flag when terminal constraint is met for the first time

    for i = 0:step:Nsim-1
        % if(mod(i,10) == 1)
        %     fprintf('Closed-loop simulation: iterate %i out of %i \n', i, Nsim)
        % end
        
        % reference signal
        yr = yrr(i+1);
        
    
    %     up_num = Tpred/deltaT -1 ;  % 7 days
        up_num = time_interval/deltaT -1  ;  
        apply_interval = 0;
        
        % Koopman MPC
        xlift = liftFun(zeta); % Lift
    
        
        %% Control every "up_num" samples
        
         
    %     if( (mod(i,up_num) == apply_interval) && (Cy*x_koop > 1e-3))  %%  t = 10 compute input, apply it to t = 11
    %         u_koop = koopmanMPC(xlift,yr); % Get control input
    % %     elseif(mod(i,10) == 1)
    %         % flag = 1;
    %     else
    % %         u_koop = koopmanMPC(xlift,yr); % Get control input
    %         if((Cy*x_koop < 1e-3))
    %             flag = 1;
    %         end
    % 
    %         u_koop = 0;
    %     end
    % 
    %     if(flag==1)
    %         u_koop = 0;
    %     end
        
        %% State update (uncertain model)
        u_koop = UU_nominal(i+1); % nominal input

        seed = (j);
        x_koop = f_ud(0,x_koop,u_koop, seed); % Update true state
        zeta = [ Cy*x_koop ; u_koop; zeta(1:end-ny-m)]; % Update delay-embedded state
            
        XX_koop = [XX_koop x_koop];
        UU_koop = [UU_koop u_koop];
        
    end
    B_record(j,:) = XX_koop(1,:);
    E_record(j,:) = XX_koop(2,:);
    Ti_record(j,:) = XX_koop(3,:);
    Tu_record(j,:) = XX_koop(4,:);

    % T_u_end_nominal(j) = find(XX_koop(4,1000:end) > desr_end * 0.8, 1, 'last') * deltaT  ;
    T_u_end_nominal(j) = find(XX_koop(4,1000:end) < desr_end * 0.8, 1) * deltaT  ;

    disp(j);

    if(isempty(ind_inf))
        ind_inf = Nsim;
    end
    
end


B_record_max = prctile(B_record, 97.5);
B_record_min = prctile(B_record, 2.5);

E_record_max = prctile(E_record, 97.5);
E_record_min = prctile(E_record, 2.5);

Ti_record_max = prctile(Ti_record, 97.5);
Ti_record_min = prctile(Ti_record, 2.5);

Tu_record_max = prctile(Tu_record, 97.5);
Tu_record_min = prctile(Tu_record, 2.5);


if(isempty(ind_inf))
    ind_inf = Nsim;
end

% u_nominal = UU_koop;


%% Plot (Output) _ Uncertain parameter
font_size = 23;

figure
% subplot(1,3,j)
grid on
% Np = 30;
% Output (y = x_2)
yyaxis left
c0 = [0.1216 0.4667 0.7059];
% p6 = patch([[0:Nsim/step]*deltaT flip([0:Nsim/step]*deltaT)], [zeros(size(B_record_min(1,(1:step:end)))),  flip(B_record_max(1,(1:step:end)))],c0,'EdgeColor',c0', 'linewidth',lw_koop, 'FaceAlpha',0.3,'handlevisibility', 'off'); hold on             %% K-MPC  B 
p66 = patch([[0:Nsim/step]*deltaT flip([0:Nsim/step]*deltaT)], [B_record_min(1,(1:step:end)),  flip(B_record_max(1,(1:step:end)))],c0,'EdgeColor',c0', 'linewidth',lw_koop, 'FaceAlpha',0.5,'handlevisibility', 'off'); hold on             %% K-MPC  B 

h = line(nan, nan, 'color', c0, 'linestyle', '-', 'linewidth', lw_koop);
ylabel('BCG (1x10^6 c.f.u)')
xlabel("Time (days)")
yyaxis right
c1 = [1 0.4980 0.0627];
c2 = [0.1725 0.6275 0.1725];
c3 = [0.8392 0.1569 0.2078];
% p7=patch([[0:Nsim/step]*deltaT flip([0:Nsim/step]*deltaT)], [zeros(size(E_record_min(1,(1:step:end)))),  flip(E_record_max(1,(1:step:end)))],c2,'EdgeColor',c2, 'linewidth',lw_koop, 'FaceAlpha',0.3,'handlevisibility', 'off'); hold on             
p77=patch([[0:Nsim/step]*deltaT flip([0:Nsim/step]*deltaT)], [E_record_min(1,(1:step:end)),  flip(E_record_max(1,(1:step:end)))],c2,'EdgeColor',c2, 'linewidth',lw_koop, 'FaceAlpha',0.5,'handlevisibility', 'off'); hold on               
pp7 = line(nan, nan, 'color', c2, 'linestyle', '-', 'linewidth', lw_koop);
hold on    %% K-MPC  E 
% p8=patch([[0:Nsim/step]*deltaT flip([0:Nsim/step]*deltaT)], [zeros(size(Ti_record_min(1,(1:step:end)))),  flip(Ti_record_max(1,(1:step:end)))],c3,'EdgeColor',c3, 'linewidth',lw_koop, 'FaceAlpha',0.3,'handlevisibility', 'off'); hold on              
p88=patch([[0:Nsim/step]*deltaT flip([0:Nsim/step]*deltaT)], [Ti_record_min(1,(1:step:end)),  flip(Ti_record_max(1,(1:step:end)))],c3,'EdgeColor',c3, 'linewidth',lw_koop, 'FaceAlpha',0.5,'handlevisibility', 'off'); hold on              

pp8 = line(nan, nan, 'color', c3, 'linestyle', '-', 'linewidth', lw_koop);
hold on    %% K-MPC  Ti 
% p5=patch([[0:Nsim/step]*deltaT flip([0:Nsim/step]*deltaT)], [zeros(size(Tu_record_min(1,(1:step:end)))),  flip(Tu_record_max(1,(1:step:end)))],c1,'EdgeColor',c1, 'linewidth',lw_koop, 'FaceAlpha',0.3,'handlevisibility', 'off'); hold on            
p55=patch([[0:Nsim/step]*deltaT flip([0:Nsim/step]*deltaT)], [Tu_record_min(1,(1:step:end)),  flip(Tu_record_max(1,(1:step:end)))],c1,'EdgeColor',c1, 'linewidth',lw_koop, 'FaceAlpha',0.5,'handlevisibility', 'off'); hold on            

pp5 = line(nan, nan, 'color', c1, 'linestyle', '-', 'linewidth', lw_koop);

ylabel("Cell Population (1x10^6)")

LEG  = legend([h, pp7, pp8, pp5 ],'K-MPC-B', 'K-MPC-E','K-MPC-Ti','K-MPC-Tu');

% xlim([0,Tmax]);
xlim([0 60]);
set(LEG,'Interpreter','latex','location','southeast')
set(LEG,'Fontsize',font_size)
% axis([0,Tmax,-0.6,0.7])
set(gca,'FontSize',font_size);
set(gcf, 'renderer', 'zbuffer')


%% Error bar (uncer par)
error_plot = 0;
if(error_plot ==1)

    font_size = 20;
    mean_velocity = zeros();
    for i = 1:1
        mean_velocity(i) = mean(T_u_end_nominal)+10; % mean velocity
    end
    std_velocity = zeros();
    for i = 1:1
        std_velocity(i) = sqrt(var(T_u_end_nominal));
    end
    
    colorstring = 'bgry';
    
    % mean_velocity = mean(T_u_end);
    
    xData = 1:1;
    bar(xData, mean_velocity,'FaceAlpha',0.5 )
    % yline(2.5,'--r','2.5 ms','FontSize',font_size, LineWidth=2);
    hold on 
    errlow = 2 * std_velocity;
    errhigh = 2 * std_velocity;
    er = errorbar(xData,mean_velocity',errlow,errhigh , '.r', "CapSize",30, 'MarkerSize',1, 'MarkerEdgeColor', 'g', 'LineWidth',2,'Markersize',3);    
    % er.Color = [0 0 0];                            
    er.LineStyle = 'none';  
    ylim([8 28])
    % set(gca,'xtick',['CDI', 'SRDI', 'MPC'])
    ylabel("Time (days)")
    % title("Settling Time")
    set(gca,'xticklabel',{''})
    set(gca,'FontSize',font_size);

end

% mean_velocity = 18.1547;
% std_velocity = 3.9363;

%% Noisy initial condition

num_sim_ini_condition = 200;


% figure

len_sim = Nsim +1 ;

B_record_ini = zeros(num_sim_ini_condition, len_sim );

E_record_ini = zeros(num_sim_ini_condition, len_sim );

Ti_record_ini = zeros(num_sim_ini_condition, len_sim );

Tu_record_ini = zeros(num_sim_ini_condition, len_sim );

T_u_end_nominal_ini = zeros(1,num_sim_ini_condition);


for j = 1:num_sim_ini_condition
    %% Initial condition for the delay-embedded state (assuming zero control in the past)
    % rng(j+1111)
    % x0 = 1*rand(4,1) ;
    if j < num_sim_ini_condition
        % x_ini = [0.1, 0.1, 0, 0.8]';
        rng(j)
        x_ini = unifrnd(x_ini / 1.1, x_ini / 0.9); %% noisy initial values  
    else
        x_ini = [0.1, 0.1, 0, 0.8]';
    end 
    
    x = x_ini;
    desr_end = 1e-2;
    % yrr = 0.8 * exp(log(desr_end/x_ini(4))/desr_time * ([1:Nsim]) * Tmax / Nsim  ); % reference

    %% 
    zeta0 = [Cy*x ; NaN(nD*(ny+m),1)];
    for i = 1:nD
        upast = zeros(m,1);

        % seed_initial = j;
        % xp = f_ud(0,x,upast,seed_initial);
        xp = f_udd(0,x,upast);

        zeta0 = [Cy*xp ; upast ; zeta0(1:end-ny-m)]; 
        x = xp;
    end
    x0 = x;
    
    x_koop = x0; x_loc = x0;
    zeta = zeta0; % Delay-embedded "state"
    
    XX_koop_ini = x0; UU_koop_ini = [];
    XX_loc = x0; UU_loc = [];
    
    
    
    %% Get Jacobian of the true dynamics (for local linearization MPC)
    % x = sym('x',[4 1]); syms u;
    % % f_ud_sym = f_ud(0,x,u,seed_initial);
    % f_ud_sym = f_udd(0,x,u);
    % u_loc = 0;
    % Jx = jacobian(f_ud_sym,x);
    % Ju = jacobian(f_ud_sym,u);
    % 
    % wasinfeas= 0;
    % ind_inf = [];
    
    
    %% Closed-loop simultion start:  MPC linearized vs koopman
    
    step = 1;
    flag = 0; % flag when terminal constraint is met for the first time

    for i = 0:step:Nsim-1
        % if(mod(i,10) == 1)
        %     fprintf('Closed-loop simulation: iterate %i out of %i \n', i, Nsim)
        % end
        
        % reference signal
        yr = yrr(i+1);
        
    
    %     up_num = Tpred/deltaT -1 ;  % 7 days
        up_num = time_interval/deltaT -1  ;  
        apply_interval = 0;
        
        
        % Koopman MPC
        xlift = liftFun(zeta); % Lift
    
        
        
        %% State update

        u_koop = UU_nominal(i+1); % nominal input

        % seed = (j);
        x_koop = f_udd(0,x_koop,u_koop); % Update true state
        zeta = [ Cy*x_koop ; u_koop; zeta(1:end-ny-m)]; % Update delay-embedded state
            
        XX_koop_ini = [XX_koop_ini x_koop];
        UU_koop_ini = [UU_koop_ini u_koop];
        
    end
    B_record_ini(j,:) = XX_koop_ini(1,:);
    E_record_ini(j,:) = XX_koop_ini(2,:);
    Ti_record_ini(j,:) = XX_koop_ini(3,:);
    Tu_record_ini(j,:) = XX_koop_ini(4,:);
    
    % find(XX_koop_ini(4,1000:end) > desr_end * 0.8, 1, 'last')
    % T_u_end_nominal_ini(j) = find(XX_koop_ini(4,1000:end) > desr_end * x_ini(4), 1, 'last' ) * deltaT  ;

    T_u_end_nominal_ini(j) = find(XX_koop_ini(4,1000:end) < desr_end * x_ini(4), 1 ) * deltaT  ;

    disp(j)
    if(isempty(ind_inf))
        ind_inf = Nsim;
    end
    
end

B_record_max_ini = prctile(B_record_ini, 97.5);
B_record_min_ini = prctile(B_record_ini, 2.5);

E_record_max_ini = prctile(E_record_ini, 97.5);
E_record_min_ini = prctile(E_record_ini, 2.5);

Ti_record_max_ini = prctile(Ti_record_ini, 97.5);
Ti_record_min_ini = prctile(Ti_record_ini, 2.5);

Tu_record_max_ini = prctile(Tu_record_ini, 97.5);
Tu_record_min_ini = prctile(Tu_record_ini, 2.5);


%% Plot (Output) _ Uncertain initial condition
font_size = 23;

figure
% subplot(1,3,j)
grid on

yyaxis left
c0 = [0.1216 0.4667 0.7059];
% p6 = patch([[0:Nsim/step]*deltaT flip([0:Nsim/step]*deltaT)], [zeros(size(B_record_min(1,(1:step:end)))),  flip(B_record_max(1,(1:step:end)))],c0,'EdgeColor',c0', 'linewidth',lw_koop, 'FaceAlpha',0.3,'handlevisibility', 'off'); hold on             %% K-MPC  B 
p66 = patch([[0:Nsim/step]*deltaT flip([0:Nsim/step]*deltaT)], [B_record_min_ini(1,(1:step:end)),  flip(B_record_max_ini(1,(1:step:end)))],c0,'EdgeColor',c0', 'linewidth',lw_koop, 'FaceAlpha',0.5,'handlevisibility', 'off'); hold on             %% K-MPC  B 

h = line(nan, nan, 'color', c0, 'linestyle', '-', 'linewidth', lw_koop);
ylabel('BCG (1x10^6 c.f.u)')
xlabel("Time (days)")
yyaxis right
c1 = [1 0.4980 0.0627];
c2 = [0.1725 0.6275 0.1725];
c3 = [0.8392 0.1569 0.2078];
% p7=patch([[0:Nsim/step]*deltaT flip([0:Nsim/step]*deltaT)], [zeros(size(E_record_min(1,(1:step:end)))),  flip(E_record_max(1,(1:step:end)))],c2,'EdgeColor',c2, 'linewidth',lw_koop, 'FaceAlpha',0.3,'handlevisibility', 'off'); hold on             
p77=patch([[0:Nsim/step]*deltaT flip([0:Nsim/step]*deltaT)], [E_record_min_ini(1,(1:step:end)),  flip(E_record_max_ini(1,(1:step:end)))],c2,'EdgeColor',c2, 'linewidth',lw_koop, 'FaceAlpha',0.5,'handlevisibility', 'off'); hold on               
pp7 = line(nan, nan, 'color', c2, 'linestyle', '-', 'linewidth', lw_koop);
hold on    %% K-MPC  E 
% p8=patch([[0:Nsim/step]*deltaT flip([0:Nsim/step]*deltaT)], [zeros(size(Ti_record_min(1,(1:step:end)))),  flip(Ti_record_max(1,(1:step:end)))],c3,'EdgeColor',c3, 'linewidth',lw_koop, 'FaceAlpha',0.3,'handlevisibility', 'off'); hold on              
p88=patch([[0:Nsim/step]*deltaT flip([0:Nsim/step]*deltaT)], [Ti_record_min_ini(1,(1:step:end)),  flip(Ti_record_max_ini(1,(1:step:end)))],c3,'EdgeColor',c3, 'linewidth',lw_koop, 'FaceAlpha',0.5,'handlevisibility', 'off'); hold on              

pp8 = line(nan, nan, 'color', c3, 'linestyle', '-', 'linewidth', lw_koop);
hold on    %% K-MPC  Ti 
% p5=patch([[0:Nsim/step]*deltaT flip([0:Nsim/step]*deltaT)], [zeros(size(Tu_record_min(1,(1:step:end)))),  flip(Tu_record_max(1,(1:step:end)))],c1,'EdgeColor',c1, 'linewidth',lw_koop, 'FaceAlpha',0.3,'handlevisibility', 'off'); hold on            
p55=patch([[0:Nsim/step]*deltaT flip([0:Nsim/step]*deltaT)], [Tu_record_min_ini(1,(1:step:end)),  flip(Tu_record_max_ini(1,(1:step:end)))],c1,'EdgeColor',c1, 'linewidth',lw_koop, 'FaceAlpha',0.5,'handlevisibility', 'off'); hold on            

pp5 = line(nan, nan, 'color', c1, 'linestyle', '-', 'linewidth', lw_koop);

% hu = line(nan, nan, 'color', c1, 'linestyle', '-', 'linewidth', lw_koop);
% huu= plot([0:Nsim/step-1]*deltaT, yrr(1:step:end),'--b','linewidth',lw_koop);  %% Reference

ylabel("Cell Population (1x10^6)")

LEG  = legend([h, pp7, pp8, pp5 ],'K-MPC-B', 'K-MPC-E','K-MPC-Ti','K-MPC-Tu');

% xlim([0,Tmax]);
xlim([0 60]);
set(LEG,'Interpreter','latex','location','southeast')
set(LEG,'Fontsize',font_size)
% axis([0,Tmax,-0.6,0.7])
set(gca,'FontSize',font_size);
set(gcf, 'renderer', 'zbuffer')

%% Error bar (noisy initial condition)
error_plot = 0;
if(error_plot ==1)


    font_size = 20;
    mean_velocity = zeros();
    for i = 1:1
        mean_velocity(i) = mean(T_u_end_nominal_ini)+10; % mean velocity
    end
    std_velocity = zeros();
    for i = 1:1
        std_velocity(i) = sqrt(var(T_u_end_nominal_ini));
    end
    
    colorstring = 'bgry';
    
    % mean_velocity = mean(T_u_end);
    
    xData = 1:1;
    bar(xData, mean_velocity,'FaceAlpha',0.5 )
    % yline(2.5,'--r','2.5 ms','FontSize',font_size, LineWidth=2);
    hold on 
    errlow = 2 * std_velocity;
    errhigh = 2 * std_velocity;
    er = errorbar(xData,mean_velocity',errlow,errhigh , '.r', "CapSize",30, 'MarkerSize',1, 'MarkerEdgeColor', 'g', 'LineWidth',2,'Markersize',3);    
    % er.Color = [0 0 0];                            
    er.LineStyle = 'none';  
    ylim([13 22])
    % set(gca,'xtick',['CDI', 'SRDI', 'MPC'])
    ylabel("Time (days)")
    set(gca,'xticklabel',{''})
    set(gca,'FontSize',font_size);

end

% mean_velocity = 17.7048
% std_velocity = 1.7857

%%
%%
%% Plot (control input)
lw_koop = 2;
mkdir('H:\Users\yrt05\Desktop\MPC')
addpath(genpath('H:\Users\yrt05\Desktop\MPC'))

load('u_plot.mat')  % "x,y"

% load(fullfile('H:\Users\yrt05\Desktop\MPC', 'u_plot.mat'))

% Control signal
figure
marker_size = 3;

% p1 = plot([0:ind_inf(1)-1]*deltaT,UU_loc(1:ind_inf(1)),'--g','linewidth',lw_koop); hold on %% L-MPC
input_log =  UU_koop(1:end);
input_log(input_log==0)=nan;  
p2 = scatter(  [0:Nsim/step-2]*deltaT  ,  input_log(1:end-1), 80, "diamond" ,"LineWidth",marker_size); hold on        %% K-MPC
nansum(input_log(1:end-1))

u_RMC = scatter(  x ,  y, 80, "o" ,"LineWidth",marker_size); hold on 
nansum(y)
% p3 = plot([0:Nsim/step-1]*deltaT, Uub * ones(Nsim/step,1)','-k','linewidth',lw_koop); %% Constraints
% p4 = plot([0:Nsim]*deltaT,0 * ones(Nsim+1,1),'-k','linewidth',lw_koop-1); hold on %% Constraints
axis([0 Tmax 0 Uub+0.5  ])
LEG  = legend([p2,u_RMC],'K-MPC input','RMC input');
grid on
xlim([0 60]);
ylim([2 7]);
% title("Control Input (BCG)")
xlabel("Time (days)")

ylabel({"BCG dose (10^6 c.f.u)"})
set(LEG,'Interpreter','latex','location','southeast')
set(gca,'FontSize',font_size);
