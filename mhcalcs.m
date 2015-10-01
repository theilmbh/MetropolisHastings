% some mh calculations
close all

omega = 0.25/0.01;
epsilon = 0.01;
test = [];
theoretical = [];
kappa = 0.25*omega^2*epsilon^2;
g = (1-kappa)/(1+kappa)
for ntstart = 1:100
    tstart = floor(200 +rand()*(600 - 200));
    
    for t2 = tstart:tstart+20
        test(ntstart, t2 - (tstart -1)) = mean(paths(:, tstart).*paths(:, t2));
        
    end
    
end
test_avg = mean(test, 1);
theor_avg = mean(theoretical, 1);
te_list = 0:0.01:20;
theoretical= ((1+kappa) / (2*epsilon*omega))*exp(-omega*((te_list)*epsilon));
plot(0:1:20, test_avg, '.', 'MarkerSize', 24);
hold on
plot(te_list, theoretical, 'LineWidth', 2);
legend('My Code', 'Theory')