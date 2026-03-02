// D3 chart utilities for "The End Game" blog post
(function(window) {
  'use strict';

  var COLORS = {
    light: {
      bg: '#fafaf9',
      text: '#1a1a2e',
      muted: '#6b6b8a',
      grid: '#e5e5e5',
      phase1: '#2563eb',
      phase2: '#d97706',
      phase3: '#db2777',
      phase4: '#7c3aed',
      marker: '#dc2626',
      accent: '#0ea5e9'
    },
    dark: {
      bg: '#18181b',
      text: '#e4e4e7',
      muted: '#71717a',
      grid: '#3f3f46',
      phase1: '#60a5fa',
      phase2: '#fbbf24',
      phase3: '#f472b6',
      phase4: '#a78bfa',
      marker: '#f87171',
      accent: '#38bdf8'
    }
  };

  function getTheme() {
    return document.documentElement.getAttribute('data-theme') || 'light';
  }

  function getColors() {
    return COLORS[getTheme()];
  }

  var tooltipEl = null;
  function tooltip() {
    if (!tooltipEl) {
      tooltipEl = d3.select('body').append('div')
        .attr('class', 'endgame-chart-tooltip')
        .style('position', 'absolute')
        .style('pointer-events', 'none')
        .style('opacity', 0)
        .style('padding', '10px 14px')
        .style('border-radius', '6px')
        .style('font-size', '13px')
        .style('line-height', '1.5')
        .style('max-width', '320px')
        .style('z-index', '1000')
        .style('box-shadow', '0 2px 8px rgba(0,0,0,0.15)')
        .style('transition', 'opacity 0.15s');
    }
    var c = getColors();
    tooltipEl
      .style('background', c.bg)
      .style('color', c.text)
      .style('border', '1px solid ' + c.muted);
    return tooltipEl;
  }

  function showTooltip(event, html) {
    var tt = tooltip();
    tt.html(html)
      .style('opacity', 1)
      .style('left', (event.pageX + 14) + 'px')
      .style('top', (event.pageY - 14) + 'px');
  }

  function moveTooltip(event) {
    var tt = tooltip();
    tt.style('left', (event.pageX + 14) + 'px')
      .style('top', (event.pageY - 14) + 'px');
  }

  function hideTooltip() {
    var tt = tooltip();
    tt.style('opacity', 0);
  }

  // ──────────────────────────────────────────────
  // Part 1: Interactive Four-Phase Timeline
  // ──────────────────────────────────────────────

  var phases = [
    {
      id: 'phase1',
      label: 'Phase I',
      subtitle: 'Efficiency Divergence',
      years: '2024\u20132027',
      start: 2024,
      end: 2027,
      color: 'phase1',
      details: [
        'Mid-tier white-collar hollowing',
        '10x\u2013100x cognitive throughput gains',
        'Senior "architects" direct agent fleets',
        'Junior roles face systemic displacement',
        'First-movers undercut on price'
      ]
    },
    {
      id: 'phase2',
      label: 'Phase II',
      subtitle: 'Commodity Plateau',
      years: '2027\u20132030',
      start: 2027,
      end: 2030,
      color: 'phase2',
      details: [
        'Zero-marginal-cost trap hits',
        '"Perfect" digital output becomes commodity',
        'Revenue collapse in SaaS & content',
        '"Human-Signed" outputs emerge',
        'Market values authenticity over perfection'
      ]
    },
    {
      id: 'phase3',
      label: 'Phase III',
      subtitle: 'Trust & Taste',
      years: '2030\u20132035',
      start: 2030,
      end: 2035,
      color: 'phase3',
      details: [
        '90% economy runs on M2M commerce',
        'Synthetic Sphere: invisible & autonomous',
        'Organic Sphere: human experience premium',
        'Accountability becomes a marketable asset',
        'Two-tier economy fully emerges'
      ]
    },
    {
      id: 'phase4',
      label: 'Phase IV',
      subtitle: 'Post-Labor',
      years: '2035+',
      start: 2035,
      end: 2040,
      color: 'phase4',
      details: [
        'Labor decoupled from survival',
        'Capital ownership drives wealth',
        '"Proof of Intent" as hard currency',
        'Human inefficiency becomes luxury',
        'Reputation economy matures'
      ]
    }
  ];

  var markers = [
    { year: 2026.15, label: 'Block layoffs (4,000 cut)', color: 'marker' },
    { year: 2026.3, label: 'Citrini "2028 GIC" report', color: 'marker' }
  ];

  function renderPhaseTimeline(containerId) {
    var container = d3.select('#' + containerId);
    container.selectAll('*').remove();
    var c = getColors();

    var margin = { top: 40, right: 30, bottom: 60, left: 30 };
    var width = 820;
    var height = 320;
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

    var x = d3.scaleLinear()
      .domain([2023, 2041])
      .range([0, innerW]);

    // Year axis
    var yearTicks = [2024, 2026, 2028, 2030, 2032, 2034, 2036, 2038, 2040];
    g.selectAll('.year-tick')
      .data(yearTicks)
      .enter().append('text')
      .attr('x', function(d) { return x(d); })
      .attr('y', innerH + 24)
      .attr('text-anchor', 'middle')
      .attr('fill', c.muted)
      .attr('font-size', '11px')
      .text(function(d) { return d; });

    // Axis line
    g.append('line')
      .attr('x1', 0).attr('x2', innerW)
      .attr('y1', innerH + 4).attr('y2', innerH + 4)
      .attr('stroke', c.grid).attr('stroke-width', 1);

    // Phase bars
    var barH = 64;
    var barY = innerH / 2 - barH / 2;

    g.selectAll('.phase-bar')
      .data(phases)
      .enter().append('rect')
      .attr('x', function(d) { return x(d.start) + 1; })
      .attr('y', barY)
      .attr('width', function(d) { return x(d.end) - x(d.start) - 2; })
      .attr('height', barH)
      .attr('rx', 6)
      .attr('fill', function(d) { return c[d.color]; })
      .attr('opacity', 0.25)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        d3.select(this).attr('opacity', 0.45);
        var html = '<strong style="color:' + c[d.color] + '">' + d.label + ': ' + d.subtitle + '</strong>' +
          '<br><span style="color:' + c.muted + '">' + d.years + '</span>' +
          '<ul style="margin:6px 0 0;padding-left:18px">' +
          d.details.map(function(t) { return '<li>' + t + '</li>'; }).join('') +
          '</ul>';
        showTooltip(event, html);
      })
      .on('mousemove', moveTooltip)
      .on('mouseout', function() {
        d3.select(this).attr('opacity', 0.25);
        hideTooltip();
      });

    // Phase labels (two lines: label + subtitle)
    phases.forEach(function(d) {
      var cx = (x(d.start) + x(d.end)) / 2;
      var cy = barY + barH / 2;

      g.append('text')
        .attr('x', cx)
        .attr('y', cy - 7)
        .attr('text-anchor', 'middle')
        .attr('fill', c.text)
        .attr('font-size', '12px')
        .attr('font-weight', '700')
        .style('pointer-events', 'none')
        .text(d.label);

      g.append('text')
        .attr('x', cx)
        .attr('y', cy + 9)
        .attr('text-anchor', 'middle')
        .attr('fill', c.text)
        .attr('font-size', '10px')
        .attr('font-weight', '400')
        .style('pointer-events', 'none')
        .text(d.subtitle);
    });

    // "Now" indicator
    var nowX = x(2026);
    g.append('line')
      .attr('x1', nowX).attr('x2', nowX)
      .attr('y1', barY - 16).attr('y2', innerH + 4)
      .attr('stroke', c.marker)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,3');

    g.append('text')
      .attr('x', nowX)
      .attr('y', barY - 22)
      .attr('text-anchor', 'middle')
      .attr('fill', c.marker)
      .attr('font-size', '12px')
      .attr('font-weight', '600')
      .text('Now (2026)');

    // Event markers
    markers.forEach(function(m) {
      var mx = x(m.year);
      g.append('circle')
        .attr('cx', mx)
        .attr('cy', innerH + 4)
        .attr('r', 5)
        .attr('fill', c[m.color])
        .style('cursor', 'pointer')
        .on('mouseover', function(event) {
          d3.select(this).attr('r', 7);
          showTooltip(event, '<strong>' + m.label + '</strong>');
        })
        .on('mousemove', moveTooltip)
        .on('mouseout', function() {
          d3.select(this).attr('r', 5);
          hideTooltip();
        });
    });

    // Title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 18)
      .attr('text-anchor', 'middle')
      .attr('fill', c.text)
      .attr('font-size', '14px')
      .attr('font-weight', '600')
      .text('The Four Phases of the AI Transition');
  }

  // ──────────────────────────────────────────────
  // Theme change redraw
  // ──────────────────────────────────────────────

  var registry = {};

  function register(containerId, renderFn) {
    registry[containerId] = renderFn;
  }

  document.documentElement.addEventListener('themechange', function() {
    if (tooltipEl) {
      var c = getColors();
      tooltipEl
        .style('background', c.bg)
        .style('color', c.text)
        .style('border', '1px solid ' + c.muted);
    }
    Object.keys(registry).forEach(function(id) {
      registry[id]();
    });
  });

  function renderAndRegister(containerId, renderFn) {
    register(containerId, function() { renderFn(containerId); });
    renderFn(containerId);
  }

  window.EndgameCharts = {
    COLORS: COLORS,
    getTheme: getTheme,
    getColors: getColors,
    renderPhaseTimeline: function(id) { renderAndRegister(id, renderPhaseTimeline); }
  };

})(window);
