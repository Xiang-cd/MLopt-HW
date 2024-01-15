

function [x, g] = test()
    x = randn(5, 1);
    gamma = 1;
    g = MCP(x, gamma);
end

function g = MCP(x, gamma)
    g_mask = abs(x) > gamma;
    f_mask = 1 - g_mask;
    g_part = g_mask * gamma;
    f_part = f_mask .* (abs(x) - x.^2 / gamma / 2);
    g = g_part + f_part;
end