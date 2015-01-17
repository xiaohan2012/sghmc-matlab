function [result_q, result_p] = hmc(U, grad_U, epsilon, L, current_q, ...
                                    MH_flag)
    % One iteration of Monto Carlo sampling using Hamiltonian
    % dynamics
    %% U, the potential function
    %% grad_U, gradient of potential function
    %% L, the leapfrog step number
    %% epslion, leapfrog step size
    %% current_q, current value for target variable
    %% MH_flag, whether to perform MH correction

    q = current_q(:);


    % independent standard normal variates
    %%%% Why not add the previous momentum status
    p = mvnrnd(zeros(length(q), 1), diag(ones(length(q), 1)));
    p = p(:);
    current_p = p;

    % Make a half step for momentum at the beginning
    p = p - epsilon * grad_U(q) / 2;

    % Alternate full steps for position and momentum
    for i = 1:L
        % Make a full step for the position
        q = q + epsilon * p;
        % Make a full step for the momentum, except at end of
        % trajectory
        if i ~= L
            p = p - epsilon * grad_U(q);
        end
    end
    % Make a half step for momentum at the end.
    p = p - epsilon * grad_U(q) / 2;

    %% Negate momentum at end of trajectory to make the proposal
    %% symmetric
    % p = -p;

    % Evaluate potential and kinetic energies at start and end of
    % trajectory
    current_U = U(current_q);
    current_K = sum(current_p .^ 2) / 2;
    proposed_U = U(q);
    proposed_K = sum(p .^ 2) / 2;

    % Accept or reject the state at end of trajectory, returning
    % either
    % the position at the end of the trajectory or the initial
    % position
    
    if MH_flag
        u = rand(); % random sample the threshold
        if u < min(1, exp(current_U - proposed_U +current_K - ...
                          proposed_K))
            % if greater than one, then accept
            % else do it randomly
            result_q = q; % accept
            result_p = p;
        else
            result_q = current_q; % reject
            result_p = current_p;
        end
    else
        result_q = q; % accept
        result_p = p;
    end
end