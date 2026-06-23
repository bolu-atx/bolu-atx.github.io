// Shared D3 chart utilities for the pipeline thinking blog series
(function(window) {
  'use strict';

  const COLORS = {
    light: {
      bg: '#fafaf9',
      text: '#1a1a2e',
      muted: '#6b6b8a',
      grid: '#e5e5e5',
      pandas: '#2563eb',
      polars: '#059669',
      duckdb: '#7c3aed',
    },
    dark: {
      bg: '#18181b',
      text: '#e4e4e7',
      muted: '#71717a',
      grid: '#3f3f46',
      pandas: '#60a5fa',
      polars: '#34d399',
      duckdb: '#a78bfa',
    }
  };

  function getTheme() {
    return document.documentElement.getAttribute('data-theme') || 'light';
  }

  function getColors() {
    return COLORS[getTheme()];
  }

  let tooltipEl = null;
  function tooltip() {
    if (!tooltipEl) {
      tooltipEl = d3.select('body').append('div')
        .attr('class', 'pipeline-chart-tooltip')
        .style('position', 'absolute')
        .style('pointer-events', 'none')
        .style('opacity', 0)
        .style('padding', '8px 12px')
        .style('border-radius', '6px')
        .style('font-size', '13px')
        .style('line-height', '1.4')
        .style('max-width', '280px')
        .style('z-index', '1000')
        .style('box-shadow', '0 2px 8px rgba(0,0,0,0.15)')
        .style('transition', 'opacity 0.15s');
    }
    const c = getColors();
    tooltipEl
      .style('background', c.bg)
      .style('color', c.text)
      .style('border', '1px solid ' + c.muted);
    return tooltipEl;
  }

  function showTooltip(event, html) {
    const tt = tooltip();
    tt.html(html)
      .style('opacity', 1)
      .style('left', (event.pageX + 12) + 'px')
      .style('top', (event.pageY - 12) + 'px');
  }

  function moveTooltip(event) {
    const tt = tooltip();
    tt.style('left', (event.pageX + 12) + 'px')
      .style('top', (event.pageY - 12) + 'px');
  }

  function hideTooltip() {
    const tt = tooltip();
    tt.style('opacity', 0);
  }

  function renderScoreboard(containerId, data) {
    var container = d3.select('#' + containerId);
    container.selectAll('*').remove();

    var c = getColors();
    var libs = ['pandas', 'polars', 'duckdb'];
    var libLabels = { pandas: 'pandas', polars: 'Polars', duckdb: 'DuckDB' };
    var yTickLabels = { 1: 'Friction', 2: 'Good', 3: 'Excellent' };

    var margin = { top: 24, right: 20, bottom: 72, left: 64 };
    var width = 600;
    var height = 340;
    var innerW = width - margin.left - margin.right;
    var innerH = height - margin.top - margin.bottom;

    var svg = container.append('svg')
      .attr('viewBox', '0 0 ' + width + ' ' + height)
      .attr('width', '100%')
      .style('max-width', width + 'px')
      .style('display', 'block')
      .style('margin', '0 auto');

    var g = svg.append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    // x0: pattern groups
    var x0 = d3.scaleBand()
      .domain(data.map(function(d) { return d.pattern; }))
      .range([0, innerW])
      .padding(0.25);

    // x1: bars within each group
    var x1 = d3.scaleBand()
      .domain(libs)
      .range([0, x0.bandwidth()])
      .padding(0.08);

    // y: score 1-3
    var y = d3.scaleLinear()
      .domain([0.5, 3.5])
      .range([innerH, 0]);

    // Grid lines
    g.selectAll('.grid-line')
      .data([1, 2, 3])
      .enter().append('line')
      .attr('x1', 0)
      .attr('x2', innerW)
      .attr('y1', function(d) { return y(d); })
      .attr('y2', function(d) { return y(d); })
      .attr('stroke', c.grid)
      .attr('stroke-dasharray', '3,3');

    // Y-axis tick labels
    g.selectAll('.y-tick-label')
      .data([1, 2, 3])
      .enter().append('text')
      .attr('x', -12)
      .attr('y', function(d) { return y(d); })
      .attr('dy', '0.35em')
      .attr('text-anchor', 'end')
      .attr('fill', c.text)
      .attr('font-size', '12px')
      .text(function(d) { return yTickLabels[d]; });

    // Bars
    var groups = g.selectAll('.bar-group')
      .data(data)
      .enter().append('g')
      .attr('transform', function(d) { return 'translate(' + x0(d.pattern) + ',0)'; });

    groups.each(function(d) {
      var group = d3.select(this);
      libs.forEach(function(lib) {
        var score = d[lib].score;
        var label = d[lib].label;
        var barY = y(score);
        var barH = innerH - barY;

        group.append('rect')
          .attr('x', x1(lib))
          .attr('y', barY)
          .attr('width', x1.bandwidth())
          .attr('height', barH)
          .attr('rx', 3)
          .attr('fill', c[lib])
          .attr('opacity', 0.85)
          .style('cursor', 'pointer')
          .on('mouseover', function(event) {
            d3.select(this).attr('opacity', 1);
            showTooltip(event,
              '<strong>' + libLabels[lib] + '</strong><br>' +
              '<span style="color:' + c.muted + '">' + d.pattern + '</span><br>' +
              label
            );
          })
          .on('mousemove', moveTooltip)
          .on('mouseout', function() {
            d3.select(this).attr('opacity', 0.85);
            hideTooltip();
          });
      });
    });

    // X-axis labels (rotated for readability)
    g.selectAll('.x-label')
      .data(data)
      .enter().append('text')
      .attr('x', function(d) { return x0(d.pattern) + x0.bandwidth() / 2; })
      .attr('y', innerH + 16)
      .attr('text-anchor', 'end')
      .attr('transform', function(d) {
        var cx = x0(d.pattern) + x0.bandwidth() / 2;
        return 'rotate(-30,' + cx + ',' + (innerH + 16) + ')';
      })
      .attr('fill', c.text)
      .attr('font-size', '11px')
      .text(function(d) { return d.pattern; });

    // Legend
    var legendY = height - 16;
    var legendItems = libs.map(function(lib, i) {
      return { lib: lib, label: libLabels[lib], x: width / 2 + (i - 1) * 100 - 30 };
    });

    legendItems.forEach(function(item) {
      svg.append('rect')
        .attr('x', item.x)
        .attr('y', legendY - 8)
        .attr('width', 12)
        .attr('height', 12)
        .attr('rx', 2)
        .attr('fill', c[item.lib]);

      svg.append('text')
        .attr('x', item.x + 16)
        .attr('y', legendY)
        .attr('dy', '0.15em')
        .attr('fill', c.text)
        .attr('font-size', '12px')
        .text(item.label);
    });
  }

  // Redraw on theme change
  var scoreboardRegistry = {};

  var origRender = renderScoreboard;
  function renderScoreboardTracked(containerId, data) {
    scoreboardRegistry[containerId] = data;
    origRender(containerId, data);
  }

  document.documentElement.addEventListener('themechange', function() {
    // Update tooltip colors on theme change
    if (tooltipEl) {
      var c = getColors();
      tooltipEl
        .style('background', c.bg)
        .style('color', c.text)
        .style('border', '1px solid ' + c.muted);
    }
    // Redraw all registered scoreboards
    Object.keys(scoreboardRegistry).forEach(function(id) {
      origRender(id, scoreboardRegistry[id]);
    });
  });

  window.PipelineCharts = {
    COLORS: COLORS,
    getTheme: getTheme,
    getColors: getColors,
    tooltip: tooltip,
    showTooltip: showTooltip,
    moveTooltip: moveTooltip,
    hideTooltip: hideTooltip,
    renderScoreboard: renderScoreboardTracked
  };

})(window);
