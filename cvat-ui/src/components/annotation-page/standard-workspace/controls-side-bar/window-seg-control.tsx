// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState, useCallback } from 'react';
import { connect } from 'react-redux';
import Icon from '@ant-design/icons';
import { LoadingOutlined } from '@ant-design/icons';
import Modal from 'antd/lib/modal';
import Text from 'antd/lib/typography/Text';
import notification from 'antd/lib/notification';

import { WindowSegIcon } from 'icons';
import {
    getCore, MLModel, ObjectState, ObjectType, ShapeType, Job, Label,
} from 'cvat-core-wrapper';
import { CombinedState } from 'reducers';
import {
    createAnnotationsAsync,
    fetchAnnotationsAsync,
} from 'actions/annotation-actions';
import CVATTooltip from 'components/common/cvat-tooltip';

const core = getCore();

const WINDOW_SEG_MODEL_NAME = 'Window Segmentation';
const WINDOW_SEG_MODEL_NAME_V2 = 'Window Segmentation + Classification';

function normalizeLabelName(name: string): string {
    return name.trim().toLowerCase().replace(/\s+/g, ' ');
}

interface StateToProps {
    detectors: MLModel[];
    jobInstance: Job;
    labels: Label[];
    frame: number;
    curZOrder: number;
    frameIsDeleted: boolean;
}

interface DispatchToProps {
    createAnnotations: (states: ObjectState[]) => Promise<void>;
    fetchAnnotations: () => Promise<void>;
}

function mapStateToProps(state: CombinedState): StateToProps {
    const {
        annotation: {
            job: { instance: jobInstance, labels },
            player: {
                frame: { number: frame, data: { deleted: frameIsDeleted } },
            },
            annotations: {
                zLayer: { cur: curZOrder },
            },
        },
        models: {
            detectors,
        },
    } = state;

    return {
        detectors,
        jobInstance: jobInstance as Job,
        labels,
        frame,
        curZOrder,
        frameIsDeleted,
    };
}

const mapDispatchToProps = {
    createAnnotations: createAnnotationsAsync,
    fetchAnnotations: fetchAnnotationsAsync,
};

type Props = StateToProps & DispatchToProps;

function WindowSegControl(props: Props): JSX.Element | null {
    const {
        detectors, jobInstance, labels, frame, curZOrder,
        createAnnotations, fetchAnnotations, frameIsDeleted,
    } = props;
    const [fetching, setFetching] = useState(false);

    const windowSegModel = detectors.find((model) => {
        const normalizedName = model.name.trim().toLowerCase();
        return model.name === WINDOW_SEG_MODEL_NAME ||
            model.name === WINDOW_SEG_MODEL_NAME_V2 ||
            normalizedName.includes('window segmentation');
    });

    const taskLabelsByName = new Map(labels.map((label) => [normalizeLabelName(label.name), label]));
    const labelMapping = windowSegModel ? Object.fromEntries(
        windowSegModel.labels
            .map((modelLabel) => {
                const taskLabel = taskLabelsByName.get(normalizeLabelName(modelLabel.name));
                if (!taskLabel) {
                    return null;
                }

                return [
                    modelLabel.name,
                    {
                        name: taskLabel.name,
                        attributes: {},
                    },
                ];
            })
            .filter((entry): entry is [string, { name: string; attributes: Record<string, string> }] => entry !== null),
    ) : {};
    const hasCompatibleLabels = Object.keys(labelMapping).length > 0;

    const handleClick = useCallback(async () => {
        if (!windowSegModel || !hasCompatibleLabels || fetching) return;

        try {
            setFetching(true);

            const result = await core.lambda.call(jobInstance.taskId, windowSegModel, {
                type: 'annotate_frame',
                frame,
                job: jobInstance.id,
                mapping: labelMapping,
                conv_mask_to_poly: false,
            }) as { version: number; shapes: any[]; tags: any[] };

            if (!result.shapes || result.shapes.length === 0) {
                notification.info({
                    message: 'No windows detected',
                    description: 'The model did not find any windows in this frame.',
                    duration: 3,
                });
                return;
            }

            const shapeStates = result.shapes.map((shape: any) => {
                const jobLabel = jobInstance.labels
                    .find((jLabel: Label) => jLabel.id === shape.label_id);

                return new core.classes.ObjectState({
                    frame,
                    objectType: ObjectType.SHAPE,
                    source: core.enums.Source.AUTO,
                    label: jobLabel,
                    shapeType: shape.type,
                    points: shape.points,
                    occluded: shape.occluded || false,
                    rotation: shape.rotation || 0,
                    zOrder: curZOrder,
                });
            });

            await createAnnotations(shapeStates);
            await fetchAnnotations();

            notification.success({
                message: 'Window segmentation complete',
                description: `Created ${shapeStates.length} annotation(s).`,
                duration: 3,
            });
        } catch (error: any) {
            notification.error({
                message: 'Window segmentation failed',
                description: error.message,
                duration: null,
            });
        } finally {
            setFetching(false);
        }
    }, [windowSegModel, hasCompatibleLabels, fetching, jobInstance, frame, curZOrder, createAnnotations, fetchAnnotations, labelMapping]);

    if (!windowSegModel) return null;

    const disabled = !hasCompatibleLabels || frameIsDeleted;
    let tooltipMessage = 'Run window segmentation on current frame';
    if (!hasCompatibleLabels) {
        tooltipMessage = 'No matching labels found between this task and the window model.';
    }

    return (
        <>
            <CVATTooltip title={tooltipMessage} placement='right'>
                <Icon
                    className={`cvat-window-seg-control ${disabled ? 'cvat-disabled-canvas-control' : ''}`}
                    component={WindowSegIcon}
                    onClick={disabled ? undefined : handleClick}
                />
            </CVATTooltip>
            {fetching && (
                <Modal
                    title='Window Segmentation'
                    zIndex={Number.MAX_SAFE_INTEGER}
                    open
                    destroyOnClose
                    closable={false}
                    footer={[]}
                >
                    <Text>Running window segmentation model...</Text>
                    <LoadingOutlined style={{ marginLeft: '10px' }} />
                </Modal>
            )}
        </>
    );
}

export default connect(mapStateToProps, mapDispatchToProps)(WindowSegControl);
