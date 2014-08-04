function [deriv] = jd(theta,ep)
fromabove=theta+ep;
frombelow=theta-ep;

deriv = 0.5*(J(fromabove) - J(frombelow))/ep;

end
