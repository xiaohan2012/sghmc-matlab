clear

U = @(x) 1/2 * x .^ 2;

grad_U = @(x) x + randn * 4;

burn_in_number = 1000;
N = 1000; 
samples = zeros(N, 2); %store \theta and r

q = rand;

epsilon = 0.1;
L = 10;

% Noisy SGHMC with MH corrections
i = 1; accept_num = 0;
while accept_num < N
    [new_q, new_p] = hmc(U, grad_U, epsilon, L, q, 1);
       
    if all(q ~= new_q) %accepted       
        if i >= burn_in_number
            accept_num = accept_num + 1;
            samples(accept_num, :) = [new_q, new_p];
        end    
        
        q = new_q;                     
    end
    i = i+1;   
end

disp('Acceptance ratio for noisy SGHMC with correction:')
disp(accept_num / (i - burn_in_number))

figure(1)

%plotting
clf

scatter(samples(:,1), samples(:,2), 'bo')

hold on

% Noisy SGHMC without MH correction
i = 1; accept_num = 0;
while accept_num < N
    [new_q, new_p] = hmc(U, grad_U, epsilon, L, q, 0);

    if all(q ~= new_q) %accepted

        if i >= burn_in_number % burn in finished
            accept_num = accept_num + 1;
            samples(accept_num, :) = [new_q, new_p]; 
        end    
        
        q = new_q;                     
    end
    i = i+1;   
end

disp('Acceptance ratio for noisy SGHMC without correction:')
disp(accept_num / (i - burn_in_number))

%plotting
scatter(samples(:,1), samples(:,2), 'ro')

hold on

% SGHMC 
i = 1; accept_num = 0;
while accept_num < N
    [new_q, new_p] = sghmc(U, grad_U, 2, epsilon, L, q, 1);

    if all(q ~= new_q) %accepted        
        if isnan(q)
            disp('Warning: is not a number for SGHMC')
            break
        end
        if i >= burn_in_number
            accept_num = accept_num + 1;            
            samples(accept_num, :) = [new_q, new_p];
        end    
        
        q = new_q;                     
    end

    i = i+1;   
end

disp('Acceptance ratio for SGHMC:')
disp(accept_num / (i - burn_in_number))

%plotting

scatter(samples(:,1), samples(:,2), 'go')


% The true probability distribution
% 
t = linspace(-pi, pi, 1000);

scatter(cos(t), -sin(t), 'yo')



l = legend('Naive SGHMC(with MH)', 'Naive SGHMC(without MH)', 'SGHMC', ...
           'Correct')

set(l,'FontSize', 10);
legendmarkeradjust(10);

xlabel('\theta')
ylabel('r')

matlab2tikz('~/Documents/hmc/divergence.tikz', 'height', '\figureheight', 'width', '\figurewidth');
%saveas(1, '~/Documents/hmc-slides/divergence.png', 'png')