img_id = 0;
nearest = [];
distances = [];
yprs = [];
for i=1:length(detect)
    id = detect(i,1);
    if id>img_id && ~isempty(distances)
        [minv, minid] = min(distances);
        detect_min_id = i - (length(distances) - minid + 1);
        nearest = [nearest; [img_id detect(detect_min_id,2:7) minv]];
        yprs = [yprs; [img_id ypr(detect_min_id,2:4)]];
        distances = [];
        img_id = img_id + 1;
    end
    ai = detect(i,2:4);
    aruco = detect(i,5:7);
    distance = norm(aruco-ai);
    distances = [distances; distance];
end
figure
subplot(6,1,1)
plot(nearest(:,1), nearest(:,2), 'r');
subplot(6,1,2)
plot(nearest(:,1), nearest(:,5), 'r');
subplot(6,1,3)
plot(nearest(:,1), nearest(:,3), 'b');
subplot(6,1,4)
plot(nearest(:,1), nearest(:,6), 'b');
% hold on
subplot(6,1,5)
plot(nearest(:,1), nearest(:,4), 'k');
% hold on
subplot(6,1,6)
plot(nearest(:,1), nearest(:,7), 'k');
hold off
legend('x/AI','x/TAG','y/AI','y/TAG','z/AI','z/TAG');
xlabel('Time sample');
ylabel('Distance (m)');
figure
plot(nearest(:,1), nearest(:,8));
xlabel('Time sample');
ylabel('Difference in estimation (m)');
figure
plot(yprs(:,1),yprs(:,2));
hold on
plot(yprs(:,1),yprs(:,3));
hold on
plot(yprs(:,1),yprs(:,4));
hold off

xlabel('Time sample');
ylabel('Euler angles (deg)');
legend('Yaw','Pitch','Roll');