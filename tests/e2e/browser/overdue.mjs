// A run that is taking longer than the backend's limit: the Simulator says so while it runs, and stops saying it after.
export default async (b, { baseUrl }) => {
  const out = {};
  await b.goto(baseUrl);
  await b.storageClear();
  await b.goto(baseUrl);
  await b.nav('Scenarios');
  await b.settle();
  out.pick = await b.click('Suez Canal Closure');
  await b.sleep(300);
  await b.settle();
  out.run = await b.click('Run through the full pipeline');
  out.overdue = await b.waitFor(/OVERDUE[^\n]*/i, 30000);
  out.whileOverdue = await b.text();
  await b.shot('overdue');
  out.finished = await b.waitFor(/PIPELINE COMPLETE[^\n]*/i, 60000);
  out.after = await b.text();
  return out;
};
