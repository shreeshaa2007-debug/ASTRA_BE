// The human in the loop, in the UI: approve, reject, and someone else deciding first.
export default async (b, { baseUrl, apiUrl, approveSim, rejectSim, conflictSim, approver }) => {
  const out = {};
  const open = async (sim) => {
    await b.goto(baseUrl);
    await b.storageSet('resilientsc:simulationId', sim);
    await b.storageRemove('resilientsc:approver');
    await b.goto(baseUrl);
    await b.nav('Compliance & Approvals');
    await b.settle();
  };

  await open(approveSim);
  out.approveBefore = await b.text();
  out.approveDisabledWithoutName = await b.click('APPROVE PLAN');
  await b.type('Your name', approver);
  await b.sleep(300);
  out.approveClick = await b.click('APPROVE PLAN');
  out.approveDone = await b.waitFor(/PLAN APPROVED[^\n]*/i, 20000);
  out.approveAfter = await b.text();
  await b.nav('Agent Monitor');
  await b.settle();
  out.monitorAfterApprove = await b.text();
  await b.nav('Overview');
  await b.settle();
  out.dashboardAfterApprove = await b.text();
  await b.shot('approval-approved');

  await open(rejectSim);
  out.rejectDisabledWithoutName = await b.click('Reject Plan');
  await b.type('Your name', approver);
  await b.sleep(300);
  out.rejectClick = await b.click('Reject Plan');
  out.rejectDone = await b.waitFor(/PLAN REJECTED[^\n]*/i, 20000);
  out.rejectAfter = await b.text();
  await b.shot('approval-rejected');

  // someone else decides while this page is still showing "pending"
  await open(conflictSim);
  out.conflictBefore = await b.text();
  const rival = `fetch(${JSON.stringify(apiUrl)} + '/api/decisions/${conflictSim}/approve', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ decided_by: 'Rival Approver' }) }).then((r) => r.status)`;
  out.rivalStatus = await b.ev(rival);
  await b.type('Your name', approver);
  await b.sleep(300);
  out.conflictClick = await b.click('APPROVE PLAN');
  out.conflictDone = await b.waitFor(/PLAN APPROVED BY RIVAL APPROVER/i, 20000);
  out.conflictAfter = await b.text();
  await b.shot('approval-conflict');
  return out;
};
