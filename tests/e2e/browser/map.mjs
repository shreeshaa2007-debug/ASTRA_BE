// The logistics map on the dashboard: what is drawn, what is labelled, what a filter does, and what a click does.
// Facts only — the Python test decides what they should be, from the API.
export default async (b, { baseUrl, simId }) => {
  const out = { views: {} };
  await b.goto(baseUrl);
  await b.storageClear();
  await b.storageSet('resilientsc:simulationId', simId);
  await b.goto(baseUrl);
  await b.settle();
  await b.sleep(2500); // the map frames its lanes with an animation
  await b.ev(`document.querySelector('.leaflet-container').scrollIntoView({ block: 'center' })`);
  await b.sleep(600);

  // everything the page can tell us about the map as it is drawn right now
  const facts = () => b.ev(`(() => {
    const map = document.querySelector('.leaflet-container').getBoundingClientRect();
    const box = (el) => { const r = el.getBoundingClientRect(); return { left: r.left, right: r.right, top: r.top, bottom: r.bottom, width: r.width, height: r.height }; };
    const hit = [...document.querySelectorAll('.leaflet-overlay-pane path')].filter((p) => p.getAttribute('stroke-opacity') === '0');
    const flowing = [...document.querySelectorAll('.leaflet-overlay-pane path.route-flow')];
    const labels = [...document.querySelectorAll('.map-label')].map((el) => ({ text: el.innerText.trim(), cls: el.className, ...box(el) }));
    const lane = flowing.length ? box(flowing[0]) : null;
    // every drawn lane (not the basemap, not the invisible hover twins): where does the picture start and end?
    const colours = ['#e0434b', '#0f9d6c', '#5b3df5', '#d9770a', '#9aa0bf']; // the COLORS in GlobalMap.tsx: disrupted, plan, alternative, delayed, normal
    const lanes = [...document.querySelectorAll('.leaflet-overlay-pane path')].filter((p) => colours.includes(p.getAttribute('stroke')) && p.getAttribute('stroke-opacity') !== '0' && p.getAttribute('stroke-dasharray'));
    // the generated lanes have dozens of vertices; the fallback for an undrawn route is a straight line with two
    const vertices = lanes.map((p) => (p.getAttribute('d').match(/L/g) || []).length);
    const extent = lanes.length ? lanes.map(box).reduce((u, r) => ({ left: Math.min(u.left, r.left), right: Math.max(u.right, r.right), top: Math.min(u.top, r.top), bottom: Math.max(u.bottom, r.bottom) })) : null;
    return {
      map: box(document.querySelector('.leaflet-container')),
      hitPaths: hit.length, flowing: flowing.length, lane, extent, minVertices: vertices.length ? Math.min(...vertices) : 0, labels,
      buttons: [...document.querySelectorAll('button')].map((x) => x.innerText.trim()).filter((t) => /^(All|Disrupted|Carrying plan)/.test(t)),
      pressed: [...document.querySelectorAll('button[aria-pressed="true"]')].map((x) => x.innerText.trim()),
      badge: [...document.querySelectorAll('button[aria-expanded]')].map((x) => x.innerText.trim()),
      oceanLabels: [...document.querySelectorAll('.ocean-label')].map((x) => x.innerText.trim()),
      legend: document.body.innerText.includes('Paths are schematic'),
    };
  })()`);

  out.views.all = await facts();
  out.listClosedAtFirst = !(await b.text()).includes('SHA-ROT-SUEZ (Shanghai');
  await b.shot('map-all');

  // the disrupted badge opens the list of routes, and closes it again
  out.badgeClick = await b.ev(`(() => { const x = document.querySelector('button[aria-expanded]'); if (!x) return 'NO_BADGE'; x.click(); return 'ok'; })()`);
  await b.sleep(300);
  out.listOpen = (await b.text()).includes('SHA-ROT-SUEZ');
  await b.ev(`document.querySelector('button[aria-expanded]').click()`);
  await b.sleep(200);
  out.listClosedAgain = !(await b.text()).includes('SHA-ROT-SUEZ (Shanghai');

  // hovering a line shows its facts
  out.tooltip = await b.ev(`(() => {
    const hit = [...document.querySelectorAll('.leaflet-overlay-pane path')].filter((p) => p.getAttribute('stroke-opacity') === '0');
    const flowing = document.querySelector('.leaflet-overlay-pane path.route-flow');
    // the hit path that lies on the plan's lane: the one whose box matches the animated line's
    const fb = flowing.getBoundingClientRect();
    const p = hit.find((h) => { const r = h.getBoundingClientRect(); return Math.abs(r.left - fb.left) < 12 && Math.abs(r.width - fb.width) < 24; });
    if (!p) return null;
    const r = p.getBoundingClientRect();
    for (const type of ['mouseover', 'mousemove']) p.dispatchEvent(new MouseEvent(type, { bubbles: true, clientX: r.left + r.width / 2, clientY: r.top + r.height / 2 }));
    return null; })()`);
  await b.sleep(400);
  out.tooltipText = await b.ev(`document.querySelector('.map-tip')?.innerText ?? null`);
  await b.ev(`(() => { const p = [...document.querySelectorAll('.leaflet-overlay-pane path')].find((x) => x.getAttribute('stroke-opacity') === '0'); p && p.dispatchEvent(new MouseEvent('mouseout', { bubbles: true })); })()`);

  // each filter
  for (const [key, label] of [['plan', 'Carrying plan'], ['disrupted', 'Disrupted ('], ['all2', 'All (']]) {
    out[key + 'Click'] = await b.click(label);
    await b.sleep(1600);
    out.views[key] = await facts();
    await b.shot('map-' + key);
  }

  // a click on a route on the dashboard opens it in Logistics
  await b.click('Carrying plan');
  await b.sleep(900);
  out.routeClick = await b.ev(`(() => {
    const flowing = document.querySelector('.leaflet-overlay-pane path.route-flow'); if (!flowing) return 'NO_LANE';
    const fb = flowing.getBoundingClientRect();
    const hit = [...document.querySelectorAll('.leaflet-overlay-pane path')].filter((p) => p.getAttribute('stroke-opacity') === '0');
    const p = hit.find((h) => { const r = h.getBoundingClientRect(); return Math.abs(r.left - fb.left) < 12 && Math.abs(r.width - fb.width) < 24; });
    if (!p) return 'NO_HIT';
    p.dispatchEvent(new MouseEvent('click', { bubbles: true })); return 'ok'; })()`);
  await b.sleep(900);
  await b.settle();
  out.afterClick = await b.text();
  await b.ev(`document.querySelector('.leaflet-container')?.scrollIntoView({ block: 'center' })`);
  await b.sleep(400);
  await b.shot('map-logistics');
  return out;
};
