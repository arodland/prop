import cartopy.crs as ccrs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.feature
from cartopy.feature.nightshade import Nightshade
matplotlib.style.use('ggplot')
import numpy as np

class Plot:
    def __init__(self, metric_name, date, decorations=True):
        self.metric_name = metric_name
        self.date = date
        self.decorations = decorations
        self.plotalpha = 0.35

        self.fig = plt.figure(figsize=(16,24))
        self.ax = plt.axes(
            projection=ccrs.PlateCarree(),
            frame_on=self.decorations,
        )
        self.ax.set_global()

        if self.decorations:
            self.ax.grid(linewidth=.5, color='black', alpha=0.25, linestyle='--')
            self.ax.set_xticks([-180, -160, -140, -120,-100, -80, -60,-40,-20, 0, 20, 40, 60,80,100, 120,140, 160,180], crs=ccrs.PlateCarree())
            self.ax.set_yticks([-80, -60,-40,-20, 0, 20, 40, 60,80], crs=ccrs.PlateCarree())
        else:
            self.ax.axis(False)
            self.ax.outline_patch.set_visible(False)
            self.ax.background_patch.set_visible(False)
            self.ax.set_xmargin(0)
            self.ax.set_ymargin(0)

        self.ax.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '110m',
            edgecolor='face',
            facecolor=np.array((0xdd,0xdd,0xcc))/256.,
            zorder=-1
            )
        )

        self.ax.add_feature(Nightshade(self.date, alpha=0.08))


    def scale_common(self):
        def map_cmap(f, cmap, lower=0, upper=1):
            out = cmap(np.linspace(lower, upper, 256))
            out = [ f(x) for x in out ]
            return matplotlib.colors.ListedColormap(out)

        cmap = plt.cm.get_cmap('viridis')
        self.cmap = map_cmap(lambda x: [x[0], x[1], x[2], 1], cmap)
        self.cmap.set_under(self.cmap(1e-5))
        self.cmap.set_over(self.cmap(1 - 1e-5))

        self.cmap_light = map_cmap(lambda x: [self.plotalpha*x[0]+(1-self.plotalpha), self.plotalpha*x[1]+(1-self.plotalpha), self.plotalpha*x[2]+(1-self.plotalpha), 1], cmap)
        self.cmap_light.set_under(self.cmap_light(1e-5))
        self.cmap_light.set_over(self.cmap_light(1 - 1e-5))

        self.cmap_dark = map_cmap(lambda x: [0.7*x[0], 0.7*x[1], 0.7*x[2], 1], cmap)
        self.cmap_dark.set_under(self.cmap_dark(1e-5))
        self.cmap_dark.set_over(self.cmap_dark(1 - 1e-5))

    def scale_generic(self):
        self.scale_common()
        self.levels = 16
        self.norm = matplotlib.colors.Normalize(clip=False)

    def scale_mufd(self):
        self.scale_common()
        self.levels = [5.3, 7, 10.1, 14, 18, 21, 24.8, 28]
        self.norm = matplotlib.colors.LogNorm(4, 35, clip=False)

    def scale_fof2(self):
        self.scale_common()
        self.levels = [1.8, 3.5, 5.3, 7, 10.1, 14]
        self.norm = matplotlib.colors.LogNorm(1.5, 15, clip=False)

    def draw_contour(self, zi, lon_min=-180, lon_max=180, lon_steps=361, lat_min=-90, lat_max=90, lat_steps=181):
        loni = np.linspace(lon_min, lon_max, lon_steps)
        lati = np.linspace(lat_min, lat_max, lat_steps)
        loni, lati = np.meshgrid(loni, lati)

        img = self.ax.imshow(
                zi,
                extent=(lon_min, lon_max, lat_min, lat_max),
                cmap=self.cmap,
                transform=ccrs.PlateCarree(),
                alpha=self.plotalpha,
                norm=self.norm
                )

        CS2 = plt.contour(
                loni, lati, zi,
                self.levels,
                cmap=self.cmap_dark,
                norm=self.norm,
                linewidths=.6,
                alpha=.75,
                transform=ccrs.PlateCarree()
                )

        prev = None
        levels = []
        for lev in CS2.levels:
            if prev is None or '%.0f'%(lev) != '%.0f'%(prev):
                levels.append(lev)
                prev = lev

        plt.clabel(CS2, levels, inline=True, fontsize=10, fmt='%.0f', use_clabeltext=True)

        if self.decorations:
            cbar = plt.colorbar(
                    plt.cm.ScalarMappable(norm=self.norm, cmap=self.cmap_light),
                    extend='both',
                    fraction=0.03,
                    orientation='horizontal',
                    pad=0.02,
                    format=matplotlib.ticker.ScalarFormatter(),
                    ticks=CS2.levels,
                    drawedges=False
                    )
            cbar.add_lines(CS2)


    def draw_title(self, metric, extra):
        plt.title(metric + ' ' + str(self.date.strftime('%Y-%m-%d %H:%M') + ' ' + extra))

    def draw_dot(self, lon, lat, text, color, alpha):
        self.ax.text(lon, lat, text,
            fontsize=9,
            ha='left',
            transform=ccrs.PlateCarree(),
            alpha=alpha,
            bbox={
                'boxstyle': 'circle',
                'alpha': alpha - 0.1,
                'color': color,
                'mutation_scale': 0.5
            }
        )

    def draw_dots(self, df, metric):
        for index, row in df.iterrows():
            lon = float(row['station.longitude'])
            lat = float(row['station.latitude'])

            self.draw_dot(lon, lat,
                text=int(row[metric] + 0.5),
                alpha=0.2 + 0.6 * row.cs,
                color=self.cmap(self.norm(row[metric])),
            )

    def write(self, filename, format=None):
        if self.decorations:
            plt.tight_layout()
            plt.savefig(filename, format=format, dpi=180, bbox_inches='tight')
        else:
            plt.tight_layout(pad=0.0)
            plt.savefig(filename, format=format, dpi=256, bbox_inches='tight', pad_inches=0.0)

