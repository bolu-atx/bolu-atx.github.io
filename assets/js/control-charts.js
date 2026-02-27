// Shared D3 chart utilities for the control systems blog post
(function(window) {
  'use strict';

  const COLORS = {
    light: {
      bg: '#fafaf9',
      text: '#1a1a2e',
      muted: '#6b6b8a',
      grid: '#e5e5e5',
      convergent: '#22c55e',
      oscillatory: '#f472b6',
      divergent: '#ef4444',
      planned: '#a78bfa',
      actual: '#60a5fa',
      horizon: '#fbbf24',
      disturbance: '#f87171',
      attractor: '#6b7280',
    },
    dark: {
      bg: '#18181b',
      text: '#e4e4e7',
      muted: '#71717a',
      grid: '#3f3f46',
      convergent: '#4ade80',
      oscillatory: '#f9a8d4',
      divergent: '#f87171',
      planned: '#c4b5fd',
      actual: '#93c5fd',
      horizon: '#fde68a',
      disturbance: '#fca5a5',
      attractor: '#9ca3af',
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
        .attr('class', 'control-chart-tooltip')
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

  window.ControlCharts = {
    COLORS: COLORS,
    getTheme: getTheme,
    getColors: getColors,
    tooltip: tooltip,
    showTooltip: showTooltip,
    moveTooltip: moveTooltip,
    hideTooltip: hideTooltip
  };

})(window);
