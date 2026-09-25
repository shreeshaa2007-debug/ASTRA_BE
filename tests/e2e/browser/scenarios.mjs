// The Scenarios page: every definition, then one run through the full pipeline into the Simulator.
export default async (b, { baseUrl, scenarios, pipelineLabel }) => {
  const out = { pages: {} };
  await b.goto(baseUrl);
  await b.storageClear();
  await b.goto(baseUrl);
  await b.nav('Scenarios');
  await b.settle();
  for (const s of scenarios) {
    out.pages[s.label] = { clicked: await b.click(s.label), modeled: s.modeled };
    await b.sleep(150);
    out.pages[s.label].immediate = await b.text(); // what is on screen the moment the choice changes, before the new comparison can have arrived
    await b.sleep(250);
    await b.settle();
    out.pages[s.label].text = await b.text();
  }
  await b.shot('scenarios-last');

  out.pipelineClick = await b.click(pipelineLabel);
  await b.sleep(300);
  await b.settle();
  out.runClick = await b.click('Run through the full pipeline');
  out.landed = await b.waitFor(/AWAITING HUMAN APPROVAL|PIPELINE COMPLETE[^\n]*|RUN FAILED/i, 60000);
  await b.sleep(1500);
  out.simulatorText = await b.text();
  out.simId = await b.storageGet('resilientsc:simulationId');
  await b.shot('scenarios-run-simulator');
  return out;
};
