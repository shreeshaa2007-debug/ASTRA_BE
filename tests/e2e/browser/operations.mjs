// The Agent Monitor's Operations panel: what it shows, and a backtest run from the page.
export default async (b, { baseUrl, simId, productId }) => {
  const out = {};
  await b.goto(baseUrl);
  await b.storageSet('resilientsc:simulationId', simId);
  await b.goto(baseUrl);
  await b.nav('Agent Monitor');
  await b.settle();
  out.panelReady = await b.waitFor(/READINESS[\s\S]*datasets/i, 20000);
  await b.sleep(500);
  out.before = await b.text();

  out.backtestClick = await b.click(`Backtest product ${productId}`);
  out.wape = await b.waitFor(/WAPE \d+%/i, 30000);
  await b.sleep(800);
  out.after = await b.text();
  await b.shot('operations-panel');
  return out;
};
