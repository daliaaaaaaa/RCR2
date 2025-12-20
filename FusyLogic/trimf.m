
function y = trimf(x, p)
    a = p(1); b = p(2); c = p(3);
    y = max(min((x-a)/(b-a), (c-x)/(c-b)), 0);
end

