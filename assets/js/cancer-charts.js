// Shared D3 chart utilities for the cancer diagnostics blog series
(function(window) {
  'use strict';

  const COLORS = {
    light: {
      bg: '#fafaf9',
      text: '#1a1a2e',
      muted: '#6b6b8a',
      grid: '#e5e5e5',
      blue: '#2563eb',
      green: '#059669',
      pink: '#db2777',
      purple: '#7c3aed',
      yellow: '#d97706',
      red: '#dc2626',
      MRD: '#2563eb',
      ECD: '#059669',
      CGP: '#7c3aed',
      HCT: '#d97706',
      'tumor-informed': '#2563eb',
      'tumor-naive': '#dc2626',
      'CRC-stool': '#2563eb',
      'CRC-blood': '#059669',
      'MCED': '#dc2626',
      'Liver': '#d97706',
      'Lung': '#7c3aed',
      'Other': '#6b6b8a',
    },
    dark: {
      bg: '#18181b',
      text: '#e4e4e7',
      muted: '#71717a',
      grid: '#3f3f46',
      blue: '#60a5fa',
      green: '#34d399',
      pink: '#f472b6',
      purple: '#a78bfa',
      yellow: '#fbbf24',
      red: '#f87171',
      MRD: '#60a5fa',
      ECD: '#34d399',
      CGP: '#a78bfa',
      HCT: '#fbbf24',
      'tumor-informed': '#60a5fa',
      'tumor-naive': '#f87171',
      'CRC-stool': '#60a5fa',
      'CRC-blood': '#34d399',
      'MCED': '#f87171',
      'Liver': '#fbbf24',
      'Lung': '#a78bfa',
      'Other': '#71717a',
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
        .attr('class', 'cancer-chart-tooltip')
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

  const STAR_PATH = 'M0,-8L2.3,-2.5L8.5,-2.5L3.5,1.5L5.3,7.5L0,4L-5.3,7.5L-3.5,1.5L-8.5,-2.5L-2.3,-2.5Z';

  window.CancerCharts = {
    COLORS, getTheme, getColors, tooltip, showTooltip, moveTooltip, hideTooltip, STAR_PATH
  };

})(window);
