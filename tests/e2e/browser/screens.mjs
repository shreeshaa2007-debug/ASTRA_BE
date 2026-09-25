// Every screen, first with no simulation and then with a finished one loaded.
export const NAV = ['Overview', 'Disruptions', 'Disruption Simulator', 'Agent Orchestration', 'AI Decisions', 'Inventory', 'Sourcing', 'Logistics', 'Compliance & Approvals', 'Scenarios', 'Agent Monitor'];

export default async (b, { baseUrl, simId }) => {
  const out = { empty: {}, withSim: {}, navResult: {} };
  await b.goto(baseUrl);
  await b.storageClear();
  await b.goto(baseUrl);
  await b.settle();
  out.header = await b.header();
  for (const t of NAV) {
    out.navResult[t] = await b.nav(t);
    await b.settle();
    out.empty[t] = await b.text();
  }
  await b.storageSet('resilientsc:simulationId', simId);
  await b.goto(baseUrl);
  await b.settle();
  out.sidebar = await b.sidebar();
  for (const t of NAV) {
    await b.nav(t);
    await b.settle();
    out.withSim[t] = await b.text();
  }
  await b.shot('screens-with-simulation');
  return out;
};
