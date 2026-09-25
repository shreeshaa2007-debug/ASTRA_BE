// Free-text reports through the Simulator (the LLM is scripted), and what the page remembers.
export default async (b, { baseUrl }) => {
  const out = {};
  const DONE = /AWAITING HUMAN APPROVAL|PIPELINE COMPLETE[^\n]*|NO DISRUPTION SENSED|RUN FAILED|SENSING ERROR|SENSING REJECTED[^\n]*/i;
  const simId = () => b.storageGet('resilientsc:simulationId');
  const run = async () => {
    const clicked = await b.click('RUN SIMULATION');
    await b.sleep(2500);
    return { clicked, done: await b.waitFor(DONE, 60000) };
  };

  await b.goto(baseUrl);
  await b.storageClear();
  await b.goto(baseUrl);
  await b.nav('Disruption Simulator');
  await b.settle();
  out.empty = await b.text();

  out.pickWeather = await b.click('Weather Chat');
  out.weather1 = await run();
  out.weatherSim1 = await simId();
  out.weatherText = await b.text();
  out.weather2 = await run(); // same report, same (still CREATED) simulation
  out.weatherSim2 = await simId();

  out.pickFire = await b.click('Supplier Fire');
  out.fire = await run();
  out.fireSim = await simId();
  out.fireText = await b.text();

  out.pickSuez = await b.click('Suez Canal Closure');
  out.suez = await run();
  out.suezSim = await simId();
  out.suezText = await b.text();
  await b.shot('freetext-suez');

  // a reload restores the simulation and its status without running anything
  await b.goto(baseUrl);
  await b.nav('Disruption Simulator');
  await b.settle();
  await b.sleep(1500);
  out.afterReload = await b.text();
  out.afterReloadSim = await simId();
  out.reloadSidebar = await b.sidebar();

  // the report box: the limit the backend enforces, and an empty report
  await b.click('Custom report');
  out.typeCustom = await b.type('Describe a supply-chain event', 'x'.repeat(4001));
  await b.sleep(300);
  out.runWhenTooLong = await b.click('RUN SIMULATION');
  out.counter = await b.text();
  await b.type('Describe a supply-chain event', '');
  await b.sleep(300);
  out.runWhenEmpty = await b.click('RUN SIMULATION');
  return out;
};
