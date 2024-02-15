function MPC_ctr = getMPC_delta_u(A, B, C, d ,Q ,R, QN ,N, Ulb, Uub, Xlb, Xub, solver, ulin, qlin, Q_delta_u)
% Returns an MPC controller MPC_ctr. The control input is then generated using the
% syntax MPC_ctr(x0,yr), where x0 is the current state and yr the reference to be tracked (can be zero)


% delta_u on input increments
% Q_delta_u = diag([q1, q2, ...]);  % Define your desired weights



%%
% [A, B, C, d ,Q ,R, QN ,N, Ulb, Uub, Xlb, Xub, solver] = deal(Alift,Blift,C,0,Q,R,QP,Np,Ulb, Uub, xlift_min, xlift_max,'qpoases');
%% With delta_u
% [A, B, C, d ,Q ,R, QN ,N, Ulb, Uub, Xlb, Xub, solver] = deal(Alift,Blift,C,0,Q,R,QP,Np,Ulb, Uub, xlift_min, xlift_max,'qpoases');
% Q_delta_u = 10

% OUTPUT:
% Koopman MPC controller MPC_ctr(x0,yr)

% Dynamics:
% x^+ = A*x + B*u + d
% y   = C*x

% Cost:
% J = (y_N - yr_N)'*Q_N*(y_N - yr_N) + sum_{i=0:N-1} [ (y_i - yr_i)'*Q*(y_i - yr_i) + u_i'*R*u_i + ulin_i'u + qlin'*y ]

% INPUTS:
% N = Np = pred horizon
% Xlb, Xub:  n x 1 or n x N matrix "xlift_min, xlift_max" only control the
% first state variable

% Ulb, Uub: m x 1 vector or m x N matrix
% ** If elements of Xlb, Xub, Ulb, Uub not present, set them to NaN or to +-Inf **

% yr - n_outputs x 1 vector or n_outputs x N matrix


% ulin - linear term in the cost
% qlin - linear term in the cost

% Example: MPC_ctr = getKoopmanMPC(A,B,C,d,Q,R,Q,N,umin, umax, xmin, xmax);
% MPC_ctr(x0,yr) generates the control input

if(~exist('solver','var') || isempty(solver))
    solver = 'qpoases'; %cplex or quadprog
end


n = size(A,1); % Number of states
m = size(B,2); % Number of control inputs


if (~exist('C','var') || isempty(C))
    C = eye(n,n);
end
p = size(C,1); % Number of outputs


x0 = zeros(n,1); % Dummy variable;   n = 13


if (~exist('Xlb','var'))
    Xlb = [];
end
if (~exist('Xub','var'))
    Xub = [];
end
if (size(Xub,2)  == 1 || size(Xlb,2)  == 1)
    if( numel(Xub) ~= n || numel(Xlb) ~= n)
        error('The dimension of Xub or Xlb seems to be wrong')
    end
    Xlb = repmat(Xlb,1,N); Xub = repmat(Xub,1,N);   % only constrain the first state in "x_lift"
end
Xub(Xub == Inf) = NaN;
Xlb(Xlb == -Inf) = NaN;

if (size(Uub,2)  == 1 || size(Ulb,2)  == 1)
    if( numel(Uub) ~= m || numel(Ulb) ~= m  )
        error('The dimension of Uub or Ulb seems to be wrong')
    end
    Ulb = repmat(Ulb,1,N); Uub = repmat(Uub,1,N);   % constraint on Input
end



% Affine term in the dynamics - handled by state inflation
if( exist('d','var') && ~isempty(d) && norm(d) ~= 0 )
    A = [A speye(n,n) ; sparse(n,n) speye(n,n)];
    B = [B ; sparse(n,m)];
    C = [C sparse(p,n)];
    if(~isempty(Xlb))
        Xlb = [Xlb ; NaN(n,N)];
    end
    if(~isempty(Xub))
        Xub = [Xub ; NaN(n,N)];
    end
    n = size(A,1);
    x0 = [x0;d];
else
    d = NaN;  % only run this line
end


% Linear term in the cost
if(~exist('ulin','var') || isempty(ulin))
    ulin = zeros(m*N,1); % only run this line  40x1
else
    if(numel(ulin) == m)
        if (size(ulin,2) > size(ulin,1))
            ulin = ulin';
        end
        ulin = repmat(ulin,N,1);
    end
end


if(~exist('qlin','var') || isempty(qlin))
    qlin = zeros(p*N,1); % only run this line
else
    warning('Functionality of qlin not tested properly!')
    if(numel(qlin) == p)
        qlin = repmat(qlin(:),N,1);
    elseif(numel(qlin) == N*p)
        qlin = qlin(:);
    else
        error('Wrong size of qlin')
    end
end

% test = B;
% 
% B = test;

%% Get "M" and "C" matrix in MPC
[Ab, Bb] = createMPCmatrices_delta(A,{B},N);
% M = Ab; C = Bb;
Bb = Bb{1};



% AA = [A B; zeros(1,15) eye(1,1)];
% BB = [B ; eye(1,1)];
% CC = [C ; zeros(1,1)];


% [~,P,~] = dlqr(A,B,diag(Q*ones(n,1)'),diag(R*ones(1,1)'));
[~,P,~] = dlqr(A , B , C'*QN*C , diag(R*ones(1,1)') ) ;
% [~,P,~] = dlqr(AA , BB , C'*QN*C , diag(R*ones(1,1)') ) ;
% schur(full(A))


% d = eig(P); % Check P positive definite
% isposdef = all(d > 0);



Qb = sparse(p*N,p*N);

% Qb(1:p*(N-1),1:p*(N-1)) = bdiag(Q,N-1); 
Qb(1:p*(N-1),1:p*(N-1)) = bdiag(0,N-1); 
% Qb(end-p+1 : end,end-p+1:end) = QN;
Qb(end-p+1 : end,end-p+1:end) =  C * P * C';

% 
% Qb(2 : end, 2:end) = bdiag(Q,N-1);
% Qb(1:1,1:1) = C * P * C'; 
Cb = bdiag(C,N);
Rb = bdiag(0,N);
Rb(1,1) = R;


% Bounds on the states
Aineq = []; bineq = [];
Xub = Xub(:); Xlb = Xlb(:);
if (~isempty(Xub))
    Aineq = [Aineq; Bb];
    bineq = [bineq; Xub - Ab*x0];
end
if (~isempty(Xlb))
    Aineq = [Aineq; -Bb];
    bineq = [bineq;-Xlb + Ab*x0];
end

Aineq = Aineq(~isnan(bineq),:);
bineq = bineq(~isnan(bineq),:);


% disp(size(Bb'*Cb'*Qb*Cb*Bb + Rb))
% H = 2*(Bb'*Cb'*Qb*Cb*Bb + Rb);
H = 2 * (Bb' * Cb' * Qb * Cb * Bb + Rb + Q_delta_u);

%% test
% test = Bb'*Cb'*Cb*Bb ;
% test = Bb'*Cb'*Qb*Cb*Bb;
% test(1,1)
% test(3,3)
% test(end,end)
% 
% H(1,1)
% H(end,end)
%%


% f = (2*x0'*Ab'*Cb'*Qb*Cb*Bb)' + ulin + Bb'*(Cb'*qlin); % Can be sped up by eleminating the transpose
f = (2 * x0' * Ab' * Cb' * Qb * Cb * Bb)' + ulin + Bb' * (Cb' * qlin + Q_delta_u * delta_u);

Ulb = Ulb(:);
Uub = Uub(:);
H = (H+H')/2; % symetrize (in case of numerical errors)


% Build the controller
M1 = 2*( (Bb'*(Cb'*Qb*Cb))*Ab );
M2 = (-2*(Qb*Cb)*Bb)';

disp('Building controller with qp-Oases')
% Solve & initialize (Here used just for initialization of the QPoases
% homothopy based QP solvers. The solution is thrown away.)
[QP,res,optval] = qpOASES_sequence( 'i',H,f,Aineq,Ulb,Uub,[],bineq );




if( all(isnan(Xlb)) )
    Xlb = [];
end
if( all(isnan(Xub)) )
    Xub = [];
end

MPC_ctr = @(x00,yrr)(mpcController_qpoases_sequence(x00,yrr, QP,N,Ab,Xlb,Xub,M1,M2,ulin,d,Ulb,Uub,p,C,Q));



end




function [U,optval,flag] = mpcController_qpoases_sequence(x0,yr, QP,N,Ab,Xlb,Xub,M1,M2,ulin,d,Ulb,Uub,p,C,Q)
% INPUTS: x0, yr

% PARAMETERS: the rest

% M1 = (2*Ab'*Cb'*Qb*Cb*Bb)';  # already in Line 199
% M2 = (-2*Qb*Cb*Bb)';


if(~exist('yr','var') || isempty(yr))
    yr = zeros(p*N,1);
    else if(size(yr,2) == 1)
            yr = repmat(yr,N,1); % Reference trajectory
         else
            yr = yr(:);
    end
end

if(~isnan(d))
    x0 = [x0 ; d];
end

% Linear part of the constratints
% Can be significantly sped up by selecting by carrying out the
% multiplication Ab*x0 only for those rows of Ab corresponding to non-nan
% entries of Xub or Xlb
bineq = [];
if (~isempty(Xub))
    bineq = [bineq; Xub - Ab*x0];
end
if (~isempty(Xlb))
    bineq = [bineq; -Xlb + Ab*x0];
end
bineq = bineq(~isnan(bineq),:);

% Linear part of the objective function
f = M1*x0 + M2*yr + ulin;
[U,optval,flag] = qpOASES_sequence( 'h',QP,f,Ulb,Uub,[],bineq );

%U = reshape(U,numel(ulin)/N,N); Return the whole predicted sequence
U = U(1:numel(ulin)/N); % Return just the first input
y = C*x0; % Should be y - yr, but yr adds just a constant term
optval = optval + y'*Q*y;


% % Linear part of the objective function
% f = M1 * x0 + M2 * (yr - C * x0) + ulin;
% H = (2 * (Bb' * Cb' * (Q + R) * Cb * Bb)); % Modified H to include R
% % Add the quadratic cost on input increments
% H = H + 2 * R;
% 
% if (all(isnan(Xlb)))
%     Xlb = [];
% end
% if (all(isnan(Xub)))
%     Xub = [];
% end
% 
% [U, optval, flag] = qpOASES_sequence('h', QP, f, Ulb, Uub, [], bineq);
% 
% % U = reshape(U, numel(ulin) / N, N); % Return the whole predicted sequence
% U = U(1:numel(ulin) / N); % Return just the first input
% y = C * x0 - yr; % Modified to include the reference
% optval = optval + y' * (Q + R) * y; % Modified to include R

end

