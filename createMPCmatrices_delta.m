% Lifted matrices for MPC on horizon N
function [Ab Bb] = createMPCmatrices_delta(A,B,N,includex0)

% [A,B,N] = deal(A,{B},N);
    if(~exist('includex0','var'))
        includex0 = 0; %first block rows of Ab and Bb corresponding to x0 removed
    end
    
    n_a = size(A,1);
    % n_b = size(B,1);
    % B = {B};

    n = n_a + size(B,2);% dim of M = [A B;0 I]

    Ab = zeros((N+1)*n, n); % display(size(Ab));
    
    
    Ab(1:n,:) = eye(n,n)  ;

    AA_mat = [A B{1} ; zeros(1, n_a) eye(1,1)]; % display(size(AA_mat)) ;

    for i = 2:N+1
%         if( size(A,1) > 1000 )
%             i
%         end
        %Ab((i-1)*n+1:i*n,:) = A^(i-1);

        Ab(  (i-1) * n +  1  :  i*n , : ) = Ab( (i-2)*n+1 : (i-1)*n , : ) * AA_mat ;
    end
    if(includex0 == 0)
        Ab = Ab(n+1:end,:); % Seems to take a lot of time if A is huge, possible to speedup
    end
    

    % Bb = zeros( n * N, N );

    
    for q = 1:length(B)  %  1:15

        % m = size(B{q},2);
        m = 1 ; % input size = 1

        Bb{q} = zeros( (N+1)*n , N*m );

        BB_mat = [ B{q} ; eye(1,1) ];  % AA_mat * [ B{q} ; eye(1,1) ]


        for i = 2:N+1
            Bb{q}(  (i-1)*n+1  :  i*n  ,  :) = AA_mat * Bb{q}(  (i-2)*n+1  :  (i-1)*n  ,  : )   ;

            Bb{q}(  (i-1)*n+1  :  n*i  ,  (i-2)*m+1  :  m*(i-1)  ) = BB_mat; %% diagonal B = 0
%%
        end
        if(includex0 == 0)
            Bb{q} = Bb{q}(n+1:end,:);
        end
        
        Bb{q}(:, 2:end) = 0;
        
    end


end