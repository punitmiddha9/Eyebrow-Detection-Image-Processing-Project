clc
clear all;
image_folder='properly_run_datasets';
filenames=dir(fullfile(image_folder,'*.jpg'))
total_images=numel(filenames);
 
for i=1:total_images
            f=fullfile(image_folder, filenames(i).name);
            I = imread(f);
            I = imresize(I, [244, 244]);
             FDetect = vision.CascadeObjectDetector;
             FaceSegment = step(FDetect,I);
             imgFace = (I(FaceSegment(1,2):FaceSegment(1,2)+FaceSegment(1,4),FaceSegment(1,1):FaceSegment(1,1)+FaceSegment(1,3),:));
             mappingLeft = FaceSegment;
             mappingRight = mappingLeft;
             
             subplot(3, 2, 1);
             imshow(imgFace); 
             hold on;
             title('Cropped image(FACE)');
              %To detect Left Eye
             EyeDetect = vision.CascadeObjectDetector('LeftEye');
             Eye=step(EyeDetect,imgFace);
             LeftEye  = Eye(1,:);
              
             %To detect Right Eye
             EyeDetect = vision.CascadeObjectDetector('RightEye');
             Eye=step(EyeDetect,imgFace);
             RightEye = Eye(2,:);
             
            %To detect Left Eyebrow
             LeftEyebrow   = LeftEye;
             LeftEyebrow(4) = (LeftEyebrow(4)/2) - 4;
             LeftEyebrow(3) = LeftEyebrow(3);
             LeftEyebrow(4) = uint8(LeftEyebrow(4));
             LeftEyebrow(3) = uint8(LeftEyebrow(3));
             mappingLeft = mappingLeft + LeftEyebrow;
             
            %To detect Right Eyebrow
             RightEyebrow  = RightEye;
             RightEyebrow(4) = (RightEyebrow(4)/2) - 4;
             RightEyebrow(3) = RightEyebrow(3);
             RightEyebrow(4) = uint8(RightEyebrow(4));
             RightEyebrow(3) = uint8(RightEyebrow(3));
             mappingRight = mappingRight + RightEyebrow;
             
             subplot(3, 2, 2);
             imshow(imgFace); 
             hold on;
             title('Detected eyes and eyebrows');
             
             for i = 1:size(LeftEye,1)
                rectangle('Position',LeftEye(i,:),'LineWidth',2,'LineStyle','-','EdgeColor','r');
             end
              
             for i = 1:size(RightEye,1)
                rectangle('Position',RightEye(i,:),'LineWidth',2,'LineStyle','-','EdgeColor','r');
             end
             
             for i = 1:size(LeftEyebrow,1)
                rectangle('Position',LeftEyebrow(i,:),'LineWidth',2,'LineStyle','-','EdgeColor','g');
             end
             
             for i = 1:size(RightEyebrow,1)
                rectangle('Position',RightEyebrow(i,:),'LineWidth',2,'LineStyle','-','EdgeColor','g');
             end
             %To show the left eyebrow as a figure:
 
             BW = processEyebrows(imgFace, LeftEyebrow); 
             
             [startlx, stoplx, startly, stoply, contourL] = findContours(BW, mappingLeft);
             
             widthL = findWidth(contourL);
             
             subplot(3, 2, 3);
             imshow(I); hold on
             plot(contourL(:,2),contourL(:,1),'g','LineWidth',1);
             plot(startlx, startly, 'o');
             plot(stoplx, stoply, 'o');
             
             distance = num2str(pdist([startlx, startly; stoplx, stoply], 'euclidean'));
             t = strcat('Distance= ', distance, '    Width=', num2str(widthL));
             title(t);
             
              BW = processEyebrows(imgFace, RightEyebrow);
             
             [startrx, stoprx, startry, stopry, contourR] = findContours(BW, mappingRight);
             
             widthR = findWidth(contourR);
             
             subplot(3,2,4);
             imshow(I); hold on;
             plot(contourR(:,2),contourR(:,1),'g','LineWidth',1);
             plot(startrx, startry, 'o');
             plot(stoprx, stopry, 'o');
             
             distance = num2str(pdist([startrx, startry; stoprx, stopry], 'euclidean'));
             t = strcat('Distance= ', distance, '    Width=', num2str(widthR));
             title(t);
             
             
             subplot(3,2,5);
             imshow(I);
             hold on;
             midPointRX = (stoprx + startrx)/2;
             midPointRY = (stopry + startry)/2;
             midPointLX = (stoplx + startlx)/2;
             midPointLY = (stoply + startly)/2;
             
             distance = num2str(pdist([midPointLX, midPointLY; midPointRX, midPointRY], 'euclidean'));
             t = strcat('Distance between centres(eyebrows) =', distance);
             title(t);
             
             plot(midPointRX, midPointRY - 4, 'o');
             plot(midPointLX, midPointRY - 4, 'o');
             
             subplot(3,2,6);
             imshow(I);
             hold on;
             
             distance = num2str(pdist([startlx, startly; stoprx, stopry], 'euclidean'));
             t = strcat('Distance between ends(eyebrows)=', distance);
             title(t);
             
             plot(startlx, startly, 'o');
             plot(stoprx, stopry, 'o');
             
            
             
             figure;
             LimgEyebrow = (imgFace(LeftEyebrow(1,2):LeftEyebrow(1,2)+LeftEyebrow(1,4),LeftEyebrow(1,1):LeftEyebrow(1,1)+LeftEyebrow(1,3),:));
             subplot(4, 3, 1);
             imshow(LimgEyebrow);
             title('Left eyebrow');
             IM1 = imcomplement(LimgEyebrow);
             subplot(4, 3, 2);
             imshow(IM1);
             se = strel('disk', 10);
             afterOpening = imopen(IM1, se);
             subplot(4, 3, 3);
             imshow(afterOpening);
             IMG = IM1 - afterOpening;
             subplot(4, 3, 4);
             imshow(IMG);
             K = imadjust(IMG, [0.1 0.20], []);
             subplot(4, 3, 5);
             imshow(K);
             level = graythresh(K);
             BW = im2bw(K, level);
             subplot(4, 3, 6);
             imshow(BW);
             BW = medfilt2(BW);
             
             RimgEyebrow = (imgFace(RightEyebrow(1,2):RightEyebrow(1,2)+RightEyebrow(1,4),RightEyebrow(1,1):RightEyebrow(1,1)+RightEyebrow(1,3),:));
             subplot(4, 3, 7);
             imshow(RimgEyebrow);
             title('Right eyebrow');
             IM2 = imcomplement(RimgEyebrow);
             subplot(4, 3, 8);
             imshow(IM2);
             se = strel('disk', 10);
             afterOpening = imopen(IM2, se);
             subplot(4, 3, 9);
             imshow(afterOpening);
             IMG2 = IM2 - afterOpening;
             subplot(4, 3, 10);
             imshow(IMG2);
             L = imadjust(IMG2, [0.1 0.20], []);
             subplot(4, 3, 11);
             imshow(L);
             level = graythresh(L);
             BW = im2bw(L, level);
             subplot(4, 3, 12);
             imshow(BW);
             BW = medfilt2(BW);
end
 
 %detect contours
function [startx, stopx, starty, stopy, contour] = findContours(BW, mapping);
 
    [m, n] = size(BW);
 
     flag=0; r=0; c=0;
     for(i=1:m)
         for(j=1:n)
             if(BW(i,j)==1)
                 r=i;
                 c=j;
                 flag=1;
                 break;
             end
         end
         if(flag==1)
             break;
         end
     end
 
     contour = bwtraceboundary(BW,[r c],'E',4,Inf,'counterclockwise');
     [s1, s2] = size(contour);
     for(i=1:s1)
        contour(i,2) = contour(i,2) + mapping(1,1) - 2;
        contour(i,1) = contour(i,1) + mapping(1,2) - 2;
     end
 
     startx = contour(1,2);
     stopx = contour(1,2);
     for(i=1:s1)
         if(startx > contour(i,2))
             startx = contour(i,2);
             starty = contour(i,1);
         end
         if(stopx < contour(i,2))
             stopx = contour(i,2);
             stopy = contour(i,1);
         end
     end
end
 
%find width of eyebrow
function width = findWidth(contour)
    MAX = max(contour);
    MIN = min(contour);
    
    y = (MAX(1) + MIN(1)) / 2;
    
    [s1, s2] = size(contour);
    distance = 0;
    for(i=1:s1)
        if(contour(i,1) < y)
            for(j=1:s1)
                if(contour(j,1) > y && contour(i,2) == contour(j,2))
                    if(distance < (contour(j,1) - contour(i,1)))
                        distance = contour(j,1) - contour(i,1);
                        disp('Distance: ' + distance);
                    end
                end
            end
        end
    end
    width = distance;
end
 
%process eyebrows for contour detection
function BW = processEyebrows(imgFace, Eyebrow);
 imgEyebrow = (imgFace(Eyebrow(1,2):Eyebrow(1,2)+Eyebrow(1,4),Eyebrow(1,1):Eyebrow(1,1)+Eyebrow(1,3),:));
 IM1 = imcomplement(imgEyebrow);
 se = strel('disk', 10);
 afterOpening = imopen(IM1, se);
 IM = IM1 - afterOpening;
 K = imadjust(IM, [0.1 0.20], []);
 level = graythresh(K);
 BW = im2bw(K, level);
 BW = medfilt2(BW);
end

