vals = linspace(-1,1,500);
h = figure('Visible','off');
for xr = vals
		for xi = vals
				plot(xr,xi, whichRoot(xr,xi));
				hold on;
		end
end
print(h,'Basin','-dpng');
close(h);
