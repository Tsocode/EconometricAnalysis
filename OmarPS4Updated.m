% ############## Load systolicData ##############
systolicsystolicData = load('systolic_reg.txt');

%################ Extract systolicData ##############
systolicPressure = systolicData(:,1);
age = systolicData(:,2);
weight = systolicData(:,3);  
smoker = systolicData(:,4);

%############## Design matrix ##############
designMatrix = [ones(size(age)) age weight smoker];

%############## OLS estimates ##############  
coefficients = designMatrix\systolicPressure;

%############## Calculate standard errors ##############
residuals = systolicPressure - designMatrix*coefficients;
meanSquaredError = residuals'*residuals / (length(systolicPressure) - length(coefficients));
standardErrors = sqrt(diag(inv(designMatrix'*designMatrix)) * meanSquaredError);

%############## Calculate test statistics ##############
tStatistics = coefficients ./ standardErrors;
pValues = 2*(1 - normcdf(abs(tStatistics)));

%############## Confidence intervals ##############
alpha = 0.05;
degreesFreedom = length(systolicPressure) - length(coefficients);
criticalValue = norminv(1-alpha/2,0,1);
confidenceIntervals = coefficients' + criticalValue*standardErrors'/sqrt(degreesFreedom);

disp('OLS Estimates');
disp('---------------------------');
disp('Variable Coef Std Err t-stat p-value [95% CI]');
disp(coefficients);
disp(standardErrors);   
disp(tStatistics);
disp(pValues);
disp(confidenceIntervals);

%############## Robust standard errors ##############
XtX = designMatrix'*designMatrix;
invXtX = inv(XtX);
robustStandardErrors = sqrt(diag(invXtX * (residuals'*residuals)));

disp(' ');
disp('Robust Estimates');
disp('---------------------------');
disp('Variable Coef Robust SE t');
disp([coefficients robustStandardErrors coefficients./robustStandardErrors]);

%############## Quadratic model ##############
designMatrix2 = [ones(size(age)) age age.^2 weight smoker];
coefficients2 = designMatrix2\systolicPressure;
residuals2 = systolicPressure - designMatrix2*coefficients2;
meanSquaredError2 = residuals2'*residuals2/(length(systolicPressure)-length(coefficients2));
standardErrors2 = sqrt(diag(inv(designMatrix2'*designMatrix2)) * meanSquaredError2);

disp(' ');
disp('Quadratic Model');
disp('---------------------------');
disp('Variable Coef SE');
disp([coefficients2 standardErrors2]);

%############## Hypothesis tests ##############
% 1. Wald test
wald = coefficients2(2:3)./standardErrors2(2:3);
waldTest = wald'*wald;
pvalWald = 1 - chi2cdf(waldTest,2);

% 2. F-test 
residualsNoQuad = systolicPressure - designMatrix*coefficients;
residualsQuad = systolicPressure - designMatrix2*coefficients2;
F = ((residualsNoQuad'*residualsNoQuad - residualsQuad'*residualsQuad)/(2)) ./ (residualsQuad'*residualsQuad/(length(systolicPressure)-length(coefficients2)));
pvalF = 1 - fcdf(F,2,length(systolicPressure)-length(coefficients2));

% 3. Rss
rssNoQuad = residualsNoQuad'*residualsNoQuad;
rssQuad = residualsQuad'*residualsQuad;  
pvalRss = 1 - fcdf((rssNoQuad - rssQuad)/(2)/(rssQuad/(length(systolicPressure)-length(coefficients2))),2,length(systolicPressure)-length(coefficients2));

% 4. R2
yty = systolicPressure'*systolicPressure;
r2NoQuad = 1 - rssNoQuad ./ repmat(yty, size(rssNoQuad));
r2Quad = 1 - rssQuad ./ repmat(yty, size(rssQuad));
pvalR2 = 1 - fcdf((r2Quad - r2NoQuad)/2/(1-r2Quad)/(length(systolicPressure)-length(coefficients2)),2,length(systolicPressure)-length(coefficients2));

disp(' '); 
disp('Hypothesis Tests for H0: beta1=beta2=0');
disp('--------------------------------------------');
disp(['Wald Test: Chi2(' num2str(2) ') = ' num2str(waldTest) ', p=' num2str(pvalWald)]);
disp(['F-Test: F(' num2str(2) ',' num2str(length(systolicPressure)-length(coefficients2)) ') = ' num2str(F) ', p=' num2str(pvalF)]);
disp(['RSS Test: F(' num2str(2) ',' num2str(length(systolicPressure)-length(coefficients2)) ') = ' num2str(F) ', p=' num2str(pvalRss)]);
disp(['R2 Test: F(' num2str(2) ',' num2str(length(systolicPressure)-length(coefficients2)) ') = ' num2str(F) ', p=' num2str(pvalR2)]);


%############## Plot residuals vs age ##############
figure; 
scatter(age, residuals);
xlabel('Age');  
ylabel('Residuals');
title('Residuals vs Age');

%############## Test for heteroscedasticity ##############
[rho,pval] = corr(age, residuals.^2); 

%############## Position text higher ##############
txt = {['Correlation = ',num2str(rho)], ['p-value = ',num2str(pval)]};
text(30, 10000, txt); 

%############## Robust standard errors ##############
robustSE = sqrt(diag(invXtX * (residuals'*residuals)));