// The backend goes away while the page is open, then comes back. `hookUrl` is the test's remote control.
export default async (b, { baseUrl, hookUrl, simId }) => {
  const out = {};
  const hook = (name) => fetch(`${hookUrl}/${name}`).then((r) => r.text());
  await b.goto(baseUrl);
  await b.storageSet('resilientsc:simulationId', simId);
  await b.goto(baseUrl);
  await b.settle();
  out.onlineHeader = await b.header();
  out.onlineDashboard = await b.text();

  out.stop = await hook('stop');
  await b.click('API Connected'); // the header's "re-check the backend" button
  await b.sleep(1500);
  out.downHeader = await b.header();
  await b.nav('Scenarios');
  await b.settle();
  await b.sleep(800);
  out.downScenarios = await b.text();
  await b.nav('Inventory');
  await b.settle();
  await b.sleep(800);
  out.downInventory = await b.text();
  await b.nav('Overview');
  await b.settle();
  await b.sleep(800);
  out.downDashboard = await b.text();
  out.downProblems = [...b.problems]; // failed requests are expected while it is down
  await b.shot('outage-down');

  out.start = await hook('start');
  await b.click('API Unreachable');
  await b.sleep(2000);
  out.upHeader = await b.header();
  out.retry = await b.click('Retry');
  await b.sleep(500);
  await b.settle();
  out.recoveredDashboard = await b.text();
  await b.shot('outage-recovered');
  b.problems.length = 0; // only what happens after recovery counts against the page
  await b.nav('Scenarios');
  await b.settle();
  out.recoveredScenarios = await b.text();
  return out;
};
