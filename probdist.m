U = @(x) -2 * x .^ 2 + x .^ 4;

grad_U = @(x) -4 * x + 4 * x .^ 3 + randn * 4;



N = 10000; 
samples = zeros(N, 1); %store \theta and r

q = rand;

epsilon = 0.1;
L = 10;

% Noisy SGHMC with MH corrections
i = 1; accept_num = 0;
while accept_num < N
    [new_q, new_p] = hmc(U, grad_U, epsilon, L, q, 1);

    if all(q ~= new_q) %accepted
        accept_num = accept_num + 1;
        samples(accept_num, :) = new_q;
        q = new_q;                     
    end
    i = i+1;   
end

disp('Acceptance ratio for noisy SGHMC with correction:')
disp(accept_num / i)

figure(1)
%plotting
clf
[f, x] = hist(samples, 50);

plot(x, f / sum( f * diff(x(1:2))), 'm*-')

hold on

% Noisy SGHMC without MH correction
i = 1; accept_num = 0;
while accept_num < N
    [new_q, new_p] = hmc(U, grad_U, epsilon, L, q, 0);

    if all(q ~= new_q) %accepted
        accept_num = accept_num + 1;
        samples(accept_num, :) = new_q;
        q = new_q;                     
    end
    i = i+1;   
end

disp('Acceptance ratio for noisy SGHMC without correction:')
disp(accept_num / i)

%plotting
[f, x] = hist(samples, 50);

plot(x, f / sum( f * diff(x(1:2))), 'rx-')

hold on

% SGHMC 
i = 1; accept_num = 0;
while accept_num < N
    [new_q, new_p] = sghmc(U, grad_U, 1, epsilon, L, q, 0);

    if all(q ~= new_q) %accepted
        accept_num = accept_num + 1;
        samples(accept_num, :) = new_q;
        if isnan(q)
            break
        end
        q = new_q;                            
    end
    i = i+1;   
end

disp('Acceptance ratio for SGHMC:')
disp(accept_num / i)

%plotting

[f, x] = hist(samples, 50);

plot(x, f / sum( f * diff(x(1:2))), 'go-')



% The true probability distribution

f = exp(-U(x));
plot(x, f / sum( f * diff(x(1:2))), 'bo-')

l = legend('Naive SGHMC(with MH)', 'Naive SGHMC(without MH)', 'SGHMC', ...
           'True distribution')

set(l,'FontSize', 12);

xlabel('\theta')
axis_data = [-2.5 2.5 0 0.8];
axis(axis_data)


matlab2tikz('~/Documents/hmc/probdist.tikz', 'height', '\figureheight', 'width', '\figurewidth');
%saveas(1, '~/Documents/hmc-slides/probdist.png', 'png') 