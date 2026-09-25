// A backend that is alive but not ready: the header says so, and the Operations panel says why.
export default async (b, { baseUrl }) => {
  const out = {};
  await b.goto(baseUrl);
  await b.storageClear();
  await b.goto(baseUrl);
  await b.settle();
  await b.sleep(1500);
  out.header = await b.header();
  out.chipTitle = await b.ev(`(document.querySelector('header button[title^="Backend reachable"]') || {}).title || null`);
  await b.nav('Agent Monitor');
  await b.settle();
  out.readiness = await b.waitFor(/NOT READY[\s\S]*?this_file_does_not_exist\.csv/i, 20000);
  out.text = await b.text();
  await b.shot('not-ready');
  return out;
};
