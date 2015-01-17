U = @(x) -2 * x .^ 2 + x .^ 4;

grad_U = @(x) -4 * x + 4 * x .^ 3 + randn * 4;



N = 100000; 
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

legend('Naive SGHMC(with MH)', 'Naive SGHMC(without MH)', 'SGHMC', ...
       'True distribution')

axis_data = [-2.5 2.5 0 1];
axis(axis_data)TRACE = 1;

mu = zeros(2, 1);
sigma = [1 .95; .95 1];
sigma_inv = inv(sigma);

U = @(x) x' * sigma_inv * x / 2;

grad_U = @(x) sigma_inv * x;

%q = [-1.5 -1.55]';
q = [0 0 ]';
samples(1, :) = q;
p = [-1 1]';

N = 10; i = 1;
samples = zeros(N, 2);

accept_num = 0;

epsilon = 0.25;
L = 1;
[new_q, new_p] = hmc(U, grad_U, epsilon, L, q, p);
while i < N
    if all(q ~= new_q) %accepted
        accept_num = accept_num + 1;
        samples(i, :) = q;
        q = new_q;
        p = new_p;
        i = i+1;
        disp(q);
    end
    [new_q, new_p] = hmc(U, grad_U, epsilon, L, q, p);
end

disp('Acceptance ratio:')
disp(accept_num / N)

axis_data = [-2 2 -2 2];

clf
subplot(1,2,1)
scatter(samples(:,1), samples(:,2), 'or')
hold on

if TRACE
    plot(samples(:,1), samples(:,2))
    
    labels = cellstr( num2str([1:N]') );
    text(samples(:,1), samples(:,2), labels, ...
         'VerticalAlignment','bottom', ...
         'HorizontalAlignment','right')
end

axis(axis_data)

subplot(1,2,2)
xs = mvnrnd(mu, sigma, 1000);
scatter(xs(:,1), xs(:,2))
axis(axis_data)