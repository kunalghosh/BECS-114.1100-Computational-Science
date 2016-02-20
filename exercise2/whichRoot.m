function [c] = whichRoot(xr, xi)
    [root_r, root_i] = run_newton_method(xr, xi);
    % disp(root_r);
    % if root_r > 0 
    %     if root_r ~= 1
    %         disp([xr,xi]);
    %     end
    % end

    if root_r == 1
        c = 'g.';
    elseif root_r < 0
       if root_i > 0 
           c = 'r.';
       else
           c = 'b.';
       end
    end
end

function result = shouldIBreak(z)
    result = 0;
    if abs(real(z)) < 0.000000000001 
        if abs(imag(z)) < 0.0000000000001
           result = 1;
        end
    end
end

function [root_r, root_i] = run_newton_method(xr,xi)
    max_iter = 0;
    old_r = xr;
    old_i = xi;
    old = complex(old_r, old_i);
    % while max_iter < 100
    for i = 1:100
       new = old - (f(old) / fdash(old)); 
       if shouldIBreak(old-new)
          if shouldIBreak(f(new))
              break;
          end
       end
       old = new;
    end
    root_r = real(new);
    root_i = imag(new);
end

% function [r,i] = f(xr,xi)
%     z = complex(xr, xi);
%     retVal = (z * z * z) - 1;
%     r = real(retVal);
%     i = imag(retVal);
% end

function [znew] = f(z)
    znew = (z * z * z) -1;
end

% function [r,i] = fdash(xr,xi)
%     z = complex(xr, xi);
%     retVal = 3 * z * z;
%     r = real(retVal);
%     i = imag(retVal);
% end

function [znew] = fdash(z)
    znew = 3 * z * z;
end
